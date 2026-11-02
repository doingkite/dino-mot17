"""Use MOT17 ground-truth boxes (train sequences, e.g. MOT17-09) to scale up the video experiment.

MOT17 has box + identity GT but no point GT, so two things can be automated:

  roi-pairs  sample same-identity box pairs at larger frame gaps (15/30/60...). The output is
             fed to ``mot17_annotate_pairs.py --roi-pairs`` so you only click points (no box
             drawing), giving many more labelled points than the original 3.
  reid       'which person in frame t+gap is the query person?' rank-1 accuracy using DINOv3
             crop embeddings (CLS, patch mean, or both) vs two baselines that use no
             appearance learning: nearest box by IoU/position (motion prior) and HSV colour
             histogram. Answers whether DINOv3 recognises the same person, at box level.

Data: MOT17 train split, folder like data/MOT17/train/MOT17-09-FRCNN with img1/ and gt/gt.txt
(https://motchallenge.net/data/MOT17/). gt.txt columns: frame, id, x, y, w, h, conf, class, visibility.

  python mot17_gt_tools.py roi-pairs --sequence-dir data/MOT17/train/MOT17-09-FRCNN --gaps 15 30 60 --per-gap 10 --output roi_pairs.json
  python mot17_annotate_pairs.py --sequence-dir data/MOT17/train/MOT17-09-FRCNN --roi-pairs roi_pairs.json --points-per-pair 3 --output pairs_gt_rois.json
  python mot17_dinov3_batch.py --sequence-dir data/MOT17/train/MOT17-09-FRCNN --pairs-json pairs_gt_rois.json --output results/mot17_gt_rois
  python mot17_gt_tools.py reid --sequence-dir data/MOT17/train/MOT17-09-FRCNN --gaps 15 30 60 120 --output results/mot17_reid
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_gt(sequence_dir, min_visibility=0.5, min_size=48):
    path = sequence_dir / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (use a MOT17 *train* sequence)")
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    boxes = defaultdict(dict)   # frame -> id -> (x, y, w, h, visibility)
    for frame, pid, x, y, w, h, conf, cls, vis in raw[:, :9]:
        if conf != 1 or int(cls) != 1 or vis < min_visibility or h < min_size:
            continue
        boxes[int(frame)][int(pid)] = (int(round(x)), int(round(y)), int(round(w)), int(round(h)), float(vis))
    if not boxes:
        raise ValueError("No usable pedestrian boxes after filtering")
    return boxes


def clip_box(box, width, height):
    x, y, w, h = box[:4]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    return [int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))]


def candidate_pairs(boxes, gap):
    out = []
    for frame, people in boxes.items():
        later = boxes.get(frame + gap)
        if not later:
            continue
        for pid in people:
            if pid in later:
                out.append((frame, frame + gap, pid))
    return out


def frame_size(sequence_dir):
    import cv2

    first = sorted((sequence_dir / "img1").glob("*.jpg"))[0]
    image = cv2.imread(str(first))
    return image.shape[1], image.shape[0]


def cmd_roi_pairs(args):
    boxes = load_gt(args.sequence_dir, args.min_visibility, args.min_size)
    width, height = frame_size(args.sequence_dir)
    rng = np.random.default_rng(args.seed)
    result = []
    for gap in args.gaps:
        options = candidate_pairs(boxes, gap)
        if not options:
            print(f"gap {gap}: no same-identity pairs")
            continue
        chosen = rng.choice(len(options), min(args.per_gap, len(options)), replace=False)
        for i in sorted(chosen):
            f0, f1, pid = options[i]
            result.append({"source_frame": f0, "target_frame": f1, "person_id": pid, "gap": gap,
                           "source_roi": clip_box(boxes[f0][pid], width, height),
                           "target_roi": clip_box(boxes[f1][pid], width, height)})
        print(f"gap {gap}: {len(options)} candidate pairs, sampled {len(chosen)}")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} ROI pairs to {args.output}")


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1)


def hsv_hist(crop):
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    return hist / max(hist.sum(), 1e-9)


class CropEmbedder:
    def __init__(self, model, size, pooling):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(self.device).eval()
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.size, self.pooling = size, pooling

    def __call__(self, crop):
        import cv2
        from PIL import Image

        from mot17_dinov3_point_match import square_letterbox

        canvas, _ = square_letterbox(crop, self.size)
        pixels = self.processor(images=Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
                                return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
        with self.torch.inference_mode():
            hidden = self.model(pixel_values=pixels.to(self.device)).last_hidden_state[0].float().cpu().numpy()
        cls, patches = hidden[0], hidden[1 + self.registers:].mean(axis=0)
        vec = {"cls": cls, "mean": patches,
               "cls_mean": np.concatenate([cls / np.linalg.norm(cls), patches / np.linalg.norm(patches)])}[self.pooling]
        return vec / max(np.linalg.norm(vec), 1e-12)


def cmd_reid(args):
    import cv2

    boxes = load_gt(args.sequence_dir, args.min_visibility, args.min_size)
    width, height = frame_size(args.sequence_dir)
    rng = np.random.default_rng(args.seed)
    embed = None if args.no_dino else CropEmbedder(args.model, args.size, args.pooling)
    frames, cache = {}, {}

    def crop(frame, pid):
        key = (frame, pid)
        if key not in cache:
            if frame not in frames:
                frames.clear()
                frames[frame] = cv2.imread(str(args.sequence_dir / "img1" / f"{frame:06d}.jpg"))
            x, y, w, h = clip_box(boxes[frame][pid], width, height)
            image = frames[frame][y:y + h, x:x + w]
            cache[key] = {"hist": hsv_hist(image), "dino": embed(image) if embed else None}
        return cache[key]

    report = {"sequence": str(args.sequence_dir), "model": None if args.no_dino else args.model,
              "pooling": args.pooling, "task": "rank-1 identity retrieval among all GT pedestrians in frame t+gap",
              "gaps": {}}
    for gap in args.gaps:
        options = [o for o in candidate_pairs(boxes, gap) if len(boxes[o[1]]) >= 2]
        if not options:
            continue
        chosen = rng.choice(len(options), min(args.queries_per_gap, len(options)), replace=False)
        hits = defaultdict(int)
        gallery_sizes = []
        for i in sorted(chosen):
            f0, f1, pid = options[i]
            gallery = sorted(boxes[f1])
            gallery_sizes.append(len(gallery))
            q = crop(f0, pid)
            box0 = boxes[f0][pid]
            motion = max(gallery, key=lambda g: (iou(box0, boxes[f1][g]),
                                                 -np.hypot(box0[0] - boxes[f1][g][0], box0[1] - boxes[f1][g][1])))
            hits["motion_iou"] += motion == pid
            colour = min(gallery, key=lambda g: cv2.compareHist(q["hist"].astype(np.float32),
                                                                crop(f1, g)["hist"].astype(np.float32),
                                                                cv2.HISTCMP_BHATTACHARYYA))
            hits["hsv_histogram"] += colour == pid
            if embed:
                best = max(gallery, key=lambda g: float(q["dino"] @ crop(f1, g)["dino"]))
                hits["DINOv3"] += best == pid
            hits["random_expected"] += 1 / len(gallery)
        n = len(chosen)
        report["gaps"][str(gap)] = {"queries": n, "mean_gallery_size": float(np.mean(gallery_sizes)),
                                    **{f"rank1_{k}": v / n for k, v in hits.items()}}
        print(f"gap {gap}: " + ", ".join(f"{k}={v / n:.3f}" for k, v in hits.items()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reid_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("roi-pairs", "reid"):
        p = sub.add_parser(name)
        p.add_argument("--sequence-dir", type=Path, required=True)
        p.add_argument("--gaps", type=int, nargs="+", default=[15, 30, 60])
        p.add_argument("--min-visibility", type=float, default=0.5)
        p.add_argument("--min-size", type=int, default=48, help="minimum box height in pixels")
        p.add_argument("--seed", type=int, default=20260510)
    sub.choices["roi-pairs"].add_argument("--per-gap", type=int, default=10)
    sub.choices["roi-pairs"].add_argument("--output", type=Path, default=Path("roi_pairs.json"))
    reid = sub.choices["reid"]
    reid.add_argument("--queries-per-gap", type=int, default=300)
    reid.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    reid.add_argument("--size", type=int, default=224)
    reid.add_argument("--pooling", choices=("cls", "mean", "cls_mean"), default="cls_mean")
    reid.add_argument("--no-dino", action="store_true", help="baselines only (CPU smoke test)")
    reid.add_argument("--output", type=Path, default=Path("results/mot17_reid"))
    args = parser.parse_args()
    {"roi-pairs": cmd_roi_pairs, "reid": cmd_reid}[args.command](args)


if __name__ == "__main__":
    main()
