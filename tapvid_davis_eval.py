"""TAP-Vid-DAVIS: find the same physical point in later video frames (replaces 3 manual MOT17 points).

TAP-Vid provides human-annotated point tracks with occlusion flags, so the MOT17 question
("where is this point on the same person N frames later?") can be measured on thousands of
points instead of three. We follow the 'query_first' protocol: each track is queried at its
first visible frame and evaluated on every later frame where it is visible, in a 256x256
frame (TAP-Vid convention, frames are stretched).

Metric: position accuracy < delta for delta in 1, 2, 4, 8, 16 px and their mean
(<delta^x_avg). Occlusion prediction is NOT evaluated, so this is not Average Jaccard.
Methods
  DINOv3   query feature from the query frame, nearest patch in each target frame
           (no temporal propagation: a pure 'is this the same point' test)
  SIFT     SIFT descriptor at the query vs dense SIFT on the target frame
  static   predict the query position (zero motion) - the baseline MOT17 lacked
Data: tapvid_davis.zip from https://github.com/google-deepmind/tapnet (tapvid_davis.pkl)

  python tapvid_davis_eval.py --pkl data/tapvid_davis/tapvid_davis.pkl --max-videos 3 --output results/tapvid_pilot
  python tapvid_davis_eval.py --pkl data/tapvid_davis/tapvid_davis.pkl --output results/tapvid_davis
"""

import argparse
import csv
import io
import json
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from dense_correspondence import DenseDino, DenseSift, sample_features, nearest_patch

DELTAS = (1, 2, 4, 8, 16)
EVAL_SIZE = 256


def decode_frames(frames):
    if len(frames) and isinstance(frames[0], (bytes, bytearray)):
        from PIL import Image
        return np.stack([np.asarray(Image.open(io.BytesIO(f)).convert("RGB")) for f in frames])
    return np.asarray(frames)


def video_arrays(entry, frames):
    """points [N,T,2] normalised (x, y) and occluded [N,T]; accepts 'occluded' or 'occlusions'
    and pixel coordinates (rescaled by frame size when values exceed 1.5)."""
    points = np.asarray(entry["points"], np.float64)
    key = "occluded" if "occluded" in entry else "occlusions"
    occluded = np.asarray(entry[key], bool)
    if np.nanmax(points) > 1.5:
        height, width = frames.shape[1:3]
        points = points / np.array([width, height], np.float64)
    return points, occluded


def load_dataset(path, max_videos=None):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    names = sorted(data)[:max_videos] if max_videos else sorted(data)
    return names, data


def gap_bucket(gap):
    for edge in (5, 15, 30, 60):
        if gap <= edge:
            return f"<= {edge}"
    return "> 60"


def summarize(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r["error"])
    out = {}
    for key, errors in sorted(groups.items()):
        e = np.asarray(errors)
        item = {"points": int(len(e))}
        for d in DELTAS:
            item[f"acc_{d}px"] = float(np.mean(e < d))
        item["delta_avg"] = float(np.mean([item[f"acc_{d}px"] for d in DELTAS]))
        item["median_error_px"] = float(np.median(e))
        out[str(key)] = item
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-tracks", type=int, default=None, help="per video, for quick pilots")
    parser.add_argument("--methods", nargs="+", default=["DINOv3", "SIFT", "static"],
                        choices=("DINOv3", "SIFT", "static"))
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--refine", choices=("none", "quadratic", "soft"), default="quadratic")
    parser.add_argument("--output", type=Path, default=Path("results/tapvid_davis"))
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names, data = load_dataset(args.pkl, args.max_videos)
    print(f"Loaded {len(names)} TAP-Vid-DAVIS videos, "
          f"{sum(np.asarray(data[n]['points']).shape[0] for n in names)} tracks")
    if args.dry_run:
        return
    dino = DenseDino(args.model, args.size, mode="stretch") if "DINOv3" in args.methods else None
    sift = DenseSift(step=2, scale=8.0) if "SIFT" in args.methods else None
    rng = np.random.default_rng(args.seed)
    rows = []
    for vi, name in enumerate(names):
        frames = decode_frames(data[name]["video"])
        points, occluded = video_arrays(data[name], frames)        # [N, T, 2] normalised x, y; [N, T]
        tracks = np.arange(points.shape[0])
        if args.max_tracks and len(tracks) > args.max_tracks:
            tracks = np.sort(rng.choice(tracks, args.max_tracks, replace=False))
        small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (EVAL_SIZE, EVAL_SIZE),
                            interpolation=cv2.INTER_AREA) for f in frames]
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]
        encoded = {}

        def enc(t):
            if t not in encoded:
                encoded[t] = dino.encode(small[t])
            return encoded[t]

        by_query = defaultdict(list)
        for n in tracks:
            visible = np.flatnonzero(~occluded[n])
            if len(visible) < 2:
                continue
            by_query[int(visible[0])].append(int(n))
        for q, track_ids in by_query.items():
            q_xy = points[track_ids, q] * EVAL_SIZE
            if dino is not None:
                fq, _, gq = enc(q)
                queries = sample_features(fq, gq.to_canvas(q_xy), dino.patch_size)
            for t in range(q + 1, len(frames)):
                vis = ~occluded[track_ids, t]
                if not vis.any():
                    continue
                gt = points[track_ids, t] * EVAL_SIZE
                preds = {}
                if dino is not None:
                    ft, vt, gtgeo = enc(t)
                    xy, _ = nearest_patch(queries, ft, vt, dino.patch_size, args.refine)
                    preds["DINOv3"] = gtgeo.to_original(xy)
                if sift is not None:
                    preds["SIFT"] = sift.match(gray[q], q_xy, gray[t])
                if "static" in args.methods:
                    preds["static"] = q_xy
                for method, pred in preds.items():
                    errors = np.linalg.norm(pred - gt, axis=1)
                    for k in np.flatnonzero(vis):
                        rows.append({"video": name, "track": track_ids[k], "method": method,
                                     "query_frame": q, "frame": t, "gap": t - q, "error": float(errors[k])})
        done = summarize([r for r in rows if r["video"] == name], lambda r: r["method"])
        print(f"{vi + 1}/{len(names)} {name}: " +
              ", ".join(f"{m} d_avg={v['delta_avg']:.3f}" for m, v in done.items()))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "points.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"dataset": str(args.pkl.resolve()), "videos": len(names), "protocol": "query_first, 256x256, visible points only",
              "model": args.model if dino else None, "size": args.size, "refine": args.refine,
              "metric": "position accuracy <delta for 1,2,4,8,16px and their mean; occlusion not evaluated (not AJ)",
              "overall": summarize(rows, lambda r: r["method"]),
              "by_frame_gap": summarize(rows, lambda r: f"{r['method']}|gap {gap_bucket(r['gap'])}"),
              "by_video": summarize(rows, lambda r: f"{r['method']}|{r['video']}")}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_frame_gap": report["by_frame_gap"]}, indent=2))


if __name__ == "__main__":
    main()
