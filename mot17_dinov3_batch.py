"""Reproducible multi-point DINOv3 matching on MOT17 frame pairs.

Pairs are manually labelled with mot17_annotate_pairs.py. Person identity/ROIs
are GIVEN; this evaluates within-person point correspondence, not MOT.
"""

import argparse
import csv
import json
from pathlib import Path

from mot17_dinov3_point_match import (crop_roi, distance_metrics, image_at_frame,
                                      load_pairs, square_letterbox, valid_patch_mask)


def summarize(rows):
    import statistics

    n = len(rows)
    if not n:
        raise ValueError("No labelled points")
    metrics = ("pck_at_005", "pck_at_010", "within_16px_model_input",
               "cycle_consistent_strict", "cycle_consistent_within_one_patch",
               "baseline_static_pck_at_010", "baseline_center_pck_at_010")
    return {"n_points": n, "n_frame_pairs": len({(r["source_frame"], r["target_frame"]) for r in rows}),
            "n_person_roi_pairs": len({(r["source_frame"], r["target_frame"],
                                         tuple(r["source_roi"]), tuple(r["target_roi"])) for r in rows}),
            "median_pixel_error_original": statistics.median(r["pixel_error_original"] for r in rows),
            "pck_denominator": "max(target ROI height, target ROI width)",
            **{key: sum(bool(r[key]) for r in rows) / n for key in metrics}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-dir", type=Path, help="MOT17 train/MOT17-09-FRCNN containing img1/")
    source.add_argument("--video", type=Path, help="MP4; verify original frame alignment before using GT boxes")
    parser.add_argument("--pairs-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/mot17_dinov3_batch"))
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--dry-run", action="store_true", help="validate annotations and image frames without loading DINOv3")
    args = parser.parse_args()
    if args.size < 224 or args.size % 16:
        parser.error("--size must be >=224 and divisible by 16")
    pairs = load_pairs(args.pairs_json)
    if args.sequence_dir and not (args.sequence_dir / "img1").is_dir():
        parser.error("--sequence-dir must contain img1/")
    if args.video and not args.video.is_file():
        parser.error("--video does not exist")

    import cv2
    import numpy as np

    frames = {}
    crops = {}

    def frame_at(number):
        if number not in frames:
            if args.sequence_dir:
                path = args.sequence_dir / "img1" / f"{number:06d}.jpg"
                frame = cv2.imread(str(path))
                if frame is None:
                    raise FileNotFoundError(f"Missing frame {path}")
            else:
                frame = image_at_frame(args.video, number)
            frames[number] = frame
        return frames[number]

    def crop_at(number, roi):
        key = (number, tuple(roi))
        if key not in crops:
            crops[key] = crop_roi(frame_at(number), roi)
        return crops[key]

    for i, pair in enumerate(pairs):
        for prefix in ("source", "target"):
            crop = crop_at(pair[f"{prefix}_frame"], pair[f"{prefix}_roi"])
            x, y = pair[f"{prefix}_point"]
            if not (0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]):
                raise ValueError(f"pair {i}: {prefix}_point outside crop")
    print(f"Validated {len(pairs)} point labels across {len(frames)} frames")
    if args.dry_run:
        return

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    patch_size = int(model.config.patch_size)
    if args.size % patch_size:
        raise ValueError("--size must be divisible by the loaded model patch size")
    registers = int(model.config.num_register_tokens)
    cache = {}

    def embedding(number, roi):
        key = (number, tuple(roi))
        if key not in cache:
            letterbox, geo = square_letterbox(crop_at(number, roi), args.size)
            pil = Image.fromarray(cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB))
            pixels = processor(images=pil, return_tensors="pt", do_resize=False,
                               do_center_crop=False)["pixel_values"].to(device)
            if tuple(pixels.shape[-2:]) != (args.size, args.size):
                raise RuntimeError("Processor unexpectedly resized the image")
            with torch.inference_mode():
                hidden = model(pixel_values=pixels).last_hidden_state
            grid = args.size // patch_size
            tokens = hidden[0, 1 + registers:, :]
            if len(tokens) != grid * grid:
                raise RuntimeError("Unexpected patch token count")
            cache[key] = (F.normalize(tokens.float(), dim=-1).reshape(grid, grid, -1), geo, letterbox)
        return cache[key]

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pair in enumerate(pairs):
        f0, f1 = pair["source_frame"], pair["target_frame"]
        r0, r1 = pair["source_roi"], pair["target_roi"]
        p0, gt = pair["source_point"], pair["target_point"]
        a, geo0, input0 = embedding(f0, r0)
        b, geo1, input1 = embedding(f1, r1)
        scale0, left0, top0, _, _ = geo0
        scale1, left1, top1, _, _ = geo1
        qx = min(a.shape[1] - 1, int((p0[0] * scale0 + left0) // patch_size))
        qy = min(a.shape[0] - 1, int((p0[1] * scale0 + top0) // patch_size))
        scores = (b @ a[qy, qx]).cpu().numpy()
        valid = valid_patch_mask(scores.shape, geo1, patch_size)
        if not valid.any():
            raise RuntimeError(f"pair {i}: no valid target patches")
        scores[~valid] = -np.inf
        by, bx = np.unravel_index(np.argmax(scores), scores.shape)
        reverse = (a @ b[by, bx]).cpu().numpy()
        reverse[~valid_patch_mask(reverse.shape, geo0, patch_size)] = -np.inf
        ry, rx = np.unravel_index(np.argmax(reverse), reverse.shape)
        crop1 = crop_at(f1, r1)
        predicted = [float(np.clip(((bx + .5) * patch_size - left1) / scale1, 0, crop1.shape[1] - 1)),
                     float(np.clip(((by + .5) * patch_size - top1) / scale1, 0, crop1.shape[0] - 1))]
        metrics = distance_metrics(predicted, gt, crop1.shape, scale1, args.size)
        # The static baseline assumes the point retains the same relative ROI position.
        crop0 = crop_at(f0, r0)
        static = [p0[0] * crop1.shape[1] / crop0.shape[1],
                  p0[1] * crop1.shape[0] / crop0.shape[0]]
        center = [(crop1.shape[1] - 1) / 2, (crop1.shape[0] - 1) / 2]
        baseline_static = distance_metrics(static, gt, crop1.shape, scale1, args.size)
        baseline_center = distance_metrics(center, gt, crop1.shape, scale1, args.size)
        row = {"index": i, "source_frame": f0, "target_frame": f1,
               "person_id": pair.get("person_id", ""), "label": pair.get("label", ""),
               "source_roi": r0, "target_roi": r1, "source_point": p0,
               "ground_truth_point": gt, "predicted_point": predicted,
               "cosine_similarity": float(scores[by, bx]), "source_patch": [qx, qy],
               "target_patch": [int(bx), int(by)],
               "cycle_consistent_strict": bool((rx, ry) == (qx, qy)),
               "cycle_consistent_within_one_patch": bool(max(abs(rx - qx), abs(ry - qy)) <= 1),
               "baseline_static_pck_at_010": baseline_static["pck_at_010"],
               "baseline_center_pck_at_010": baseline_center["pck_at_010"],
               **metrics}
        rows.append(row)
        left, right = input0.copy(), input1.copy()
        pt0 = (round(p0[0] * scale0 + left0), round(p0[1] * scale0 + top0))
        pt1 = (round(predicted[0] * scale1 + left1), round(predicted[1] * scale1 + top1))
        ptgt = (round(gt[0] * scale1 + left1), round(gt[1] * scale1 + top1))
        cv2.circle(left, pt0, 8, (0, 0, 255), -1)
        cv2.circle(right, pt1, 8, (0, 255, 0), -1)
        cv2.circle(right, ptgt, 10, (255, 0, 0), 2)
        cv2.imwrite(str(args.output / f"match_{i:04d}.jpg"), np.concatenate((left, right), axis=1))
        finite = scores[np.isfinite(scores)]
        norm = np.where(np.isfinite(scores), scores, finite.min())
        norm = (norm - finite.min()) / max(float(finite.max() - finite.min()), 1e-6)
        heatmap = cv2.applyColorMap(cv2.resize((norm * 255).astype("uint8"),
                                                (args.size, args.size)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(input1, .55, heatmap, .45, 0)
        cv2.circle(overlay, pt1, 8, (0, 255, 0), 2)
        cv2.circle(overlay, ptgt, 10, (255, 0, 0), 2)
        cv2.imwrite(str(args.output / f"heatmap_{i:04d}.jpg"), overlay)
        print(f"{i + 1}/{len(pairs)}: error={metrics['pixel_error_original']:.1f}px PCK@.1={metrics['pck_at_010']}")

    summary = {"input": str((args.sequence_dir or args.video).resolve()),
               "annotations": str(args.pairs_json.resolve()), "model": args.model,
               "input_size": args.size, "patch_size": patch_size,
               "device": device, "claim": "manual point correspondence conditioned on given same-person ROIs",
               "warning": "Cosine is not probability. MOT17 GT boxes do not provide point ground truth.",
               **summarize(rows)}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "points.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
