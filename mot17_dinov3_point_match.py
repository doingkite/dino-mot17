r"""MOT17 two-frame DINOv3 patch correspondence, with manual same-person ROIs.

Run from PowerShell in a separate DINOv3-MOT17 project directory:
    & $py mot17_dinov3_point_match.py --video "$HOME\Downloads\MOT17-09-FRCNN-raw.mp4" --t0 2 --t1 2.5

The two manually selected boxes identify the SAME person. This experiment
evaluates point matching conditioned on known identity, not autonomous MOT.
"""

import argparse
import json
from pathlib import Path


def parse_box(text):
    values = tuple(int(v.strip()) for v in text.split(","))
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return values


def parse_point(text):
    values = tuple(int(v.strip()) for v in text.split(","))
    if len(values) != 2:
        raise argparse.ArgumentTypeError("point must be x,y in the source ROI")
    return values


def image_at(video_path, seconds):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS is unavailable")
    frame_number = round(seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} at {seconds}s")
    return frame, frame_number, fps


def image_at_frame(video_path, frame_1based):
    """Read a requested frame; extracted img1 frames are preferable for GT alignment."""
    import cv2

    if frame_1based < 1:
        raise ValueError("frame numbers start at 1")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1based - 1)
        ok, frame = cap.read()
        actual = round(cap.get(cv2.CAP_PROP_POS_FRAMES))
    finally:
        cap.release()
    if not ok or actual != frame_1based:
        raise RuntimeError(f"Could not verify frame {frame_1based}; use MOT17 img1 frames")
    return frame


def choose_roi(frame, title):
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, 1100 / width, 720 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    print(f"{title}: draw a box around the SAME person's upper body or bag; ENTER to confirm")
    box = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if box[2] <= 0 or box[3] <= 0:
        raise RuntimeError("ROI selection cancelled")
    x, y, w, h = (round(v / scale) for v in box)
    x, y = min(max(0, x), width - 1), min(max(0, y), height - 1)
    w, h = min(w, width - x), min(h, height - y)
    if w <= 0 or h <= 0:
        raise RuntimeError("Selected ROI has no pixels")
    return x, y, w, h


def crop_roi(frame, box):
    x, y, w, h = box
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {box} is outside the {width}x{height} frame")
    return frame[y:y + h, x:x + w].copy()


def choose_point(crop, label="FIRST crop"):
    import cv2

    height, width = crop.shape[:2]
    scale = min(3.0, 900 / width, 700 / height)
    preview = cv2.resize(crop, (round(width * scale), round(height * scale)))
    clicked = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked[:] = [(min(width - 1, round(x / scale)), min(height - 1, round(y / scale)))]

    title = "Click a distinctive point, then press ENTER"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    print(f"Click a distinct point on clothing/bag in the {label}, then ENTER")
    while True:
        canvas = preview.copy()
        if clicked:
            px, py = clicked[0]
            cv2.circle(canvas, (round(px * scale), round(py * scale)), 7, (0, 0, 255), -1)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and clicked:
            break
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("Point selection cancelled")
    cv2.destroyWindow(title)
    return clicked[0]


def square_letterbox(crop, size):
    """Resize without changing aspect ratio; retain reversible pixel geometry."""
    import cv2

    h, w = crop.shape[:2]
    scale = min(size / w, size / h)
    out_w, out_h = min(size, round(w * scale)), min(size, round(h * scale))
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    left = (size - out_w) // 2
    top = (size - out_h) // 2
    canvas = cv2.copyMakeBorder(
        resized, top, size - out_h - top, left, size - out_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return canvas, (scale, left, top, out_w, out_h)


def valid_patch_mask(shape, geometry, patch_size):
    import numpy as np

    _, left, top, valid_w, valid_h = geometry
    ys, xs = np.indices(shape)
    cx, cy = (xs + .5) * patch_size, (ys + .5) * patch_size
    return ((cx >= left) & (cx < left + valid_w) &
            (cy >= top) & (cy < top + valid_h))


def distance_metrics(prediction, truth, crop_shape, target_scale, input_size):
    import math

    dx, dy = prediction[0] - truth[0], prediction[1] - truth[1]
    original = math.hypot(dx, dy)
    # The actual model input is letterboxed with the same x/y scale.
    model = original * target_scale
    # Use a target-box scale, explicitly defined as max(height, width).
    normalized = original / max(crop_shape[:2])
    return {
        "pixel_error_original": original,
        "pixel_error_model_input": model,
        "pck_normalized_error": normalized,
        "within_16px_model_input": model <= 16,
        "within_32px_model_input": model <= 32,
        "pck_at_005": normalized <= .05,
        "pck_at_010": normalized <= .10,
    }


def load_pairs(path):
    """Strictly validate annotated pairs before loading a heavy model."""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("--pairs-json must contain a nonempty JSON array")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"pair {i}: expected an object")
        for key in ("source_frame", "target_frame", "source_roi", "target_roi",
                    "source_point", "target_point"):
            if key not in item:
                raise ValueError(f"pair {i}: missing {key}")
        for key in ("source_frame", "target_frame"):
            if type(item[key]) is not int or item[key] < 1:
                raise ValueError(f"pair {i}: {key} must be a positive integer")
        if item["source_frame"] == item["target_frame"]:
            raise ValueError(f"pair {i}: source and target must differ")
        for key, length in (("source_roi", 4), ("target_roi", 4),
                            ("source_point", 2), ("target_point", 2)):
            value = item[key]
            if (not isinstance(value, list) or len(value) != length or
                    any(type(v) is not int for v in value)):
                raise ValueError(f"pair {i}: {key} needs {length} integer values")
        if item["source_roi"][2] <= 0 or item["source_roi"][3] <= 0 or item["target_roi"][2] <= 0 or item["target_roi"][3] <= 0:
            raise ValueError(f"pair {i}: ROI width/height must be positive")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--t0", type=float, default=2.0, help="source time in seconds")
    parser.add_argument("--t1", type=float, default=2.5, help="target time in seconds")
    parser.add_argument("--size", type=int, default=448, help="multiple of patch size 16")
    parser.add_argument("--source-roi", type=parse_box, help="x,y,width,height in original frame")
    parser.add_argument("--target-roi", type=parse_box, help="x,y,width,height in original frame")
    parser.add_argument("--point", type=parse_point, help="x,y inside source ROI")
    parser.add_argument("--gt-point", type=parse_point, help="ground truth x,y inside target ROI")
    parser.add_argument("--output", type=Path, default=Path("results/mot17_dinov3_demo"))
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video file does not exist: {args.video}")
    if args.size < 224 or args.size % 16:
        parser.error("--size must be >=224 and divisible by 16")

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    first, first_idx, fps = image_at(args.video, args.t0)
    second, second_idx, _ = image_at(args.video, args.t1)
    if first_idx == second_idx:
        parser.error("t0 and t1 select the same frame")
    box0 = args.source_roi or choose_roi(first, "Frame A: source ROI")
    box1 = args.target_roi or choose_roi(second, "Frame B: same person ROI")
    crop0, crop1 = crop_roi(first, box0), crop_roi(second, box1)
    point = args.point or choose_point(crop0)
    print("Choose the SAME physical point in frame B (ground truth), before seeing the prediction")
    if not (0 <= point[0] < crop0.shape[1] and 0 <= point[1] < crop0.shape[0]):
        parser.error("--point must be inside --source-roi")
    gt_point = args.gt_point or choose_point(crop1, "SECOND crop (ground truth)")
    if not (0 <= point[0] < crop0.shape[1] and 0 <= point[1] < crop0.shape[0]):
        parser.error("--point must be inside --source-roi")
    if not (0 <= gt_point[0] < crop1.shape[1] and 0 <= gt_point[1] < crop1.shape[0]):
        parser.error("--gt-point must be inside --target-roi")

    input0, geo0 = square_letterbox(crop0, args.size)
    input1, geo1 = square_letterbox(crop1, args.size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; frame A={first_idx}; frame B={second_idx}; fps={fps:g}")
    print("Loading gated DINOv3 weights; use the HF account already granted access")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    patch_size = int(model.config.patch_size)
    registers = int(model.config.num_register_tokens)

    def features(image_bgr):
        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        # We already resized/padded the image; only model-specific rescale/normalize remain.
        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        pixels = inputs["pixel_values"].to(device)
        h, w = pixels.shape[-2:]
        if (h, w) != (args.size, args.size):
            raise RuntimeError(f"Processor unexpectedly changed size to {h}x{w}")
        with torch.inference_mode():
            hidden = model(pixel_values=pixels).last_hidden_state
        grid_h, grid_w = h // patch_size, w // patch_size
        local = hidden[0, 1 + registers:, :]
        if local.shape[0] != grid_h * grid_w:
            raise RuntimeError(f"Unexpected token count: {hidden.shape}")
        return F.normalize(local.float(), dim=-1).reshape(grid_h, grid_w, -1)

    a, b = features(input0), features(input1)
    print(f"patch grid={a.shape[0]}x{a.shape[1]}, feature width={a.shape[-1]}")
    scale0, left0, top0, _, _ = geo0
    px = point[0] * scale0 + left0
    py = point[1] * scale0 + top0
    qx = int(np.clip(px // patch_size, 0, a.shape[1] - 1))
    qy = int(np.clip(py // patch_size, 0, a.shape[0] - 1))
    query = a[qy, qx]
    similarities = (b @ query).cpu().numpy()

    # Reject letterbox-only patches. Boundaries can still partly include padding.
    scale1, left1, top1, valid_w, valid_h = geo1
    valid = valid_patch_mask(similarities.shape, geo1, patch_size)
    similarities[~valid] = -np.inf
    if not np.any(valid):
        raise RuntimeError("No valid target patches; choose a larger ROI")
    best_y, best_x = np.unravel_index(np.argmax(similarities), similarities.shape)
    score = float(similarities[best_y, best_x])
    reverse = (a @ b[best_y, best_x]).cpu().numpy()
    reverse[~valid_patch_mask(reverse.shape, geo0, patch_size)] = -np.inf
    reverse_y, reverse_x = np.unravel_index(np.argmax(reverse), reverse.shape)
    cycle_ok = (int(reverse_x), int(reverse_y)) == (qx, qy)
    cycle_within_one = max(abs(int(reverse_x) - qx), abs(int(reverse_y) - qy)) <= 1
    target_x = float(np.clip(((best_x + 0.5) * patch_size - left1) / scale1, 0, crop1.shape[1] - 1))
    target_y = float(np.clip(((best_y + 0.5) * patch_size - top1) / scale1, 0, crop1.shape[0] - 1))
    source_global = (box0[0] + point[0], box0[1] + point[1])
    target_global = (box1[0] + target_x, box1[1] + target_y)
    metrics = distance_metrics((target_x, target_y), gt_point, crop1.shape, scale1, args.size)

    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "frame_a.jpg"), first)
    cv2.imwrite(str(args.output / "frame_b.jpg"), second)
    left, right = input0.copy(), input1.copy()
    p0 = (round(point[0] * scale0 + left0), round(point[1] * scale0 + top0))
    p1 = (round(target_x * scale1 + left1), round(target_y * scale1 + top1))
    p_gt = (round(gt_point[0] * scale1 + left1), round(gt_point[1] * scale1 + top1))
    cv2.circle(left, p0, 9, (0, 0, 255), -1)
    cv2.circle(right, p1, 9, (0, 255, 0), -1)
    cv2.circle(right, p_gt, 12, (255, 0, 0), 2)
    pair = np.concatenate([left, right], axis=1)
    cv2.line(pair, p0, (p1[0] + args.size, p1[1]), (0, 255, 255), 2)
    cv2.putText(pair, f"cosine={score:.3f} cycle={cycle_ok}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(pair, "red=query green=pred blue=GT", (10, args.size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.imwrite(str(args.output / "match.jpg"), pair)
    shown = np.where(np.isfinite(similarities), similarities, np.nan)
    low, high = float(np.nanmin(shown)), float(np.nanmax(shown))
    normalized = np.nan_to_num((shown - low) / max(high - low, 1e-6), nan=0.0)
    heatmap = cv2.applyColorMap(cv2.resize((normalized * 255).astype(np.uint8),
                                             (args.size, args.size)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(input1, 0.55, heatmap, 0.45, 0)
    pixel_valid = cv2.resize(valid.astype("uint8"), (args.size, args.size),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay[~pixel_valid] = input1[~pixel_valid]
    cv2.circle(overlay, p1, 9, (0, 255, 0), 2)
    cv2.circle(overlay, p_gt, 12, (255, 0, 0), 2)
    cv2.imwrite(str(args.output / "similarity_heatmap.jpg"), overlay)
    result = {
        "video": str(args.video.resolve()), "model": args.model,
        "source_frame": first_idx, "target_frame": second_idx, "fps": fps,
        "source_roi_xywh": box0, "target_roi_xywh": box1,
        "source_point_original_xy": source_global,
        "predicted_point_original_xy": target_global,
        "ground_truth_point_original_xy": [box1[0] + gt_point[0], box1[1] + gt_point[1]],
        **metrics,
        "cosine_similarity": score, "cycle_consistent_strict": cycle_ok,
        "cycle_consistent_within_one_patch": cycle_within_one,
        "source_patch_xy": [qx, qy], "target_patch_xy": [int(best_x), int(best_y)],
        "patch_size": patch_size, "input_size": args.size,
        "warning": "ROIs and point ground truth were selected manually; cosine is not a calibrated probability. PCK denominator=max(target ROI height,width)."
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
