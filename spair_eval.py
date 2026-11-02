"""SPair-71k semantic correspondence: can DINOv3 find the SAME PART on a DIFFERENT object?

This is the counterpart to HPatches. HPatches asks for pixel-exact geometry on the same
planar scene (SIFT's home ground); SPair-71k asks for the same semantic keypoint (e.g. a
cat's left ear tip) across different instances, poses and backgrounds, where local
gradients are unreliable.

Metric: PCK@alpha = fraction of keypoints whose prediction lies within
alpha * max(target bbox width, height) of the annotation (standard SPair protocol).
Methods
  DINOv3     bilinear source feature -> most similar target patch (+ optional refine)
  SIFT       SIFT descriptor at the query vs dense SIFT on the target (hand-crafted)
  bboxprior  same relative position inside the target box (no image content at all)
Data: http://cvlab.postech.ac.kr/research/SPair-71k/  (unpack to SPair-71k/ with
JPEGImages/, PairAnnotation/test/*.json)

  python spair_eval.py --data data/SPair-71k --split test --max-pairs 300 --output results/spair_pilot
  python spair_eval.py --data data/SPair-71k --split test --output results/spair_test
"""

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np

from dense_correspondence import DenseDino, DenseSift, sample_features, nearest_patch

ALPHAS = (0.05, 0.10, 0.15)


def load_pairs(root, split, max_pairs=None, categories=None):
    files = sorted((root / "PairAnnotation" / split).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotations under {root / 'PairAnnotation' / split}")
    pairs = []
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if categories and item["category"] not in categories:
            continue
        src, trg = np.asarray(item["src_kps"], float), np.asarray(item["trg_kps"], float)
        if src.shape != trg.shape or src.ndim != 2 or not len(src):
            continue
        item["file"] = path.name
        pairs.append(item)
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def image_path(root, item, side):
    return root / "JPEGImages" / item["category"] / item[f"{side}_imname"]


def bbox_prior(src_kps, src_box, trg_box):
    sx0, sy0, sx1, sy1 = src_box
    tx0, ty0, tx1, ty1 = trg_box
    rel = (src_kps - [sx0, sy0]) / np.maximum([sx1 - sx0, sy1 - sy0], 1)
    return rel * [tx1 - tx0, ty1 - ty0] + [tx0, ty0]


def box_region(box, geo, grid, patch_size):
    x0, y0 = geo.to_canvas([[box[0], box[1]]])[0]
    x1, y1 = geo.to_canvas([[box[2], box[3]]])[0]
    centres = (np.arange(grid) + .5) * patch_size
    cols = (centres >= x0) & (centres <= x1)
    rows = (centres >= y0) & (centres <= y1)
    return rows[:, None] & cols[None, :]


class LRU(OrderedDict):
    def __init__(self, limit=64):
        super().__init__()
        self.limit = limit

    def get_or(self, key, factory):
        if key in self:
            self.move_to_end(key)
            return self[key]
        value = factory()
        self[key] = value
        if len(self) > self.limit:
            self.popitem(last=False)
        return value


def summarize(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for key, group in sorted(groups.items()):
        item = {"keypoints": len(group), "pairs": len({r["pair"] for r in group})}
        for a in ALPHAS:
            item[f"pck_{a:.2f}"] = float(np.mean([r[f"ok_{a:.2f}"] for r in group]))
            per_pair = defaultdict(list)
            for r in group:
                per_pair[r["pair"]].append(r[f"ok_{a:.2f}"])
            item[f"pck_{a:.2f}_per_pair_mean"] = float(np.mean([np.mean(v) for v in per_pair.values()]))
        out[str(key)] = item
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("trn", "val", "test"))
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--methods", nargs="+", default=["DINOv3", "SIFT", "bboxprior"],
                        choices=("DINOv3", "SIFT", "bboxprior"))
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--refine", choices=("none", "quadratic", "soft"), default="none")
    parser.add_argument("--restrict-to-bbox", action="store_true",
                        help="search only inside the target box (an easier, commonly reported variant)")
    parser.add_argument("--output", type=Path, default=Path("results/spair"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = load_pairs(args.data, args.split, args.max_pairs, args.categories)
    missing = [p for p in pairs[:50] for s in ("src", "trg") if not image_path(args.data, p, s).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images, e.g. {image_path(args.data, missing[0], 'src')}")
    print(f"Validated {len(pairs)} SPair-71k {args.split} pairs, "
          f"{sum(len(p['src_kps']) for p in pairs)} keypoints")
    if args.dry_run:
        return
    dino = DenseDino(args.model, args.size) if "DINOv3" in args.methods else None
    sift = DenseSift() if "SIFT" in args.methods else None
    images, features = LRU(128), LRU(128)
    rows = []
    for index, item in enumerate(pairs):
        load = lambda side: images.get_or(str(image_path(args.data, item, side)),  # noqa: E731
                                          lambda: cv2.imread(str(image_path(args.data, item, side))))
        src_img, trg_img = load("src"), load("trg")
        src_kps = np.asarray(item["src_kps"], float)
        trg_kps = np.asarray(item["trg_kps"], float)
        tx0, ty0, tx1, ty1 = item["trg_bndbox"]
        threshold = max(tx1 - tx0, ty1 - ty0)
        predictions = {}
        if dino is not None:
            fs, vs, gs = features.get_or(str(image_path(args.data, item, "src")), lambda: dino.encode(src_img))
            ft, vt, gt = features.get_or(str(image_path(args.data, item, "trg")), lambda: dino.encode(trg_img))
            queries = sample_features(fs, gs.to_canvas(src_kps), dino.patch_size)
            region = box_region(item["trg_bndbox"], gt, dino.grid, dino.patch_size) if args.restrict_to_bbox else None
            xy, _ = nearest_patch(queries, ft, vt, dino.patch_size, args.refine, region=region)
            predictions["DINOv3"] = gt.to_original(xy)
        if sift is not None:
            mask = None
            if args.restrict_to_bbox:
                mask = np.zeros(trg_img.shape[:2], np.uint8)
                mask[int(ty0):int(ty1) + 1, int(tx0):int(tx1) + 1] = 1
            predictions["SIFT"] = sift.match(cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY), src_kps,
                                             cv2.cvtColor(trg_img, cv2.COLOR_BGR2GRAY), mask)
        if "bboxprior" in args.methods:
            predictions["bboxprior"] = bbox_prior(src_kps, item["src_bndbox"], item["trg_bndbox"])
        for method, pred in predictions.items():
            errors = np.linalg.norm(pred - trg_kps, axis=1) / threshold
            for k, e in enumerate(errors):
                row = {"pair": item["file"], "category": item["category"], "method": method, "kp": k,
                       "viewpoint": item.get("viewpoint_variation"), "scale": item.get("scale_variation"),
                       "error_norm": float(e)}
                row.update({f"ok_{a:.2f}": bool(e <= a) for a in ALPHAS})
                rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(pairs):
            done = [r for r in rows]
            line = ", ".join(f"{m}={np.mean([r['ok_0.10'] for r in done if r['method'] == m]):.3f}"
                             for m in predictions)
            print(f"{index + 1}/{len(pairs)} PCK@0.10 so far: {line}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "keypoints.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"dataset": str(args.data.resolve()), "split": args.split, "pairs": len(pairs),
              "model": args.model if dino else None, "size": args.size, "refine": args.refine,
              "restrict_to_bbox": args.restrict_to_bbox,
              "metric": "PCK@alpha with alpha * max(target bbox w, h); micro over keypoints and mean over pairs",
              "overall": summarize(rows, lambda r: r["method"]),
              "by_category": summarize(rows, lambda r: f"{r['method']}|{r['category']}"),
              "by_viewpoint_variation": summarize(rows, lambda r: f"{r['method']}|vp{r['viewpoint']}")}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
