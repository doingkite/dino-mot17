"""Run single-variable sweeps of hpatches_spatial_eval.py and collect one comparison table.

Each configuration changes ONE factor from the v1 baseline (ViT-S/16, 448px, RANSAC 3px,
no refinement). Finished configurations are skipped, so the sweep can be resumed.

  python hpatches_sweep.py --data data/hpatches/hpatches-sequences-release --preset core
  python hpatches_sweep.py --data ... --preset all --dry-run      # print commands only

Presets
  ransac      DINOv3 RANSAC threshold 3 / 6 / 8 / 12 / 16 canvas px
  resolution  input 448 / 672 / 896 (28 / 42 / 56 patch grid)
  refine      none / quadratic / soft sub-patch refinement
  models      ViT-S/16, ViT-B/16, ViT-L/16 (needs access to each gated checkpoint)
  hybrid      DINOguidedSIFT and SIFTdinoFilter vs SIFT
  illum       i_* with target shift / homography perturbation (fair illumination test)
  learned     LoFTR and DISK+LightGlue (pip install kornia)
  core        sift + ransac + resolution + refine + hybrid + illum
  all         core + models + learned
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

MODELS = {"S": "facebook/dinov3-vits16-pretrain-lvd1689m",
          "B": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "L": "facebook/dinov3-vitl16-pretrain-lvd1689m"}


def presets():
    p = {}
    p["sift"] = [("v_SIFT_448", ["--methods", "SIFT"])]
    p["ransac"] = [(f"v_DINO_r{t}", ["--methods", "DINOv3", "--dino-ransac-threshold", str(t)])
                   for t in (3, 6, 8, 12, 16)]
    p["resolution"] = [(f"v_DINO_s{s}", ["--methods", "DINOv3", "--size", str(s)]) for s in (448, 672, 896)]
    p["refine"] = [(f"v_DINO_refine_{m}", ["--methods", "DINOv3", "--refine", m])
                   for m in ("none", "quadratic", "soft")]
    p["models"] = [(f"v_DINO_vit{k}", ["--methods", "DINOv3", "--model", name]) for k, name in MODELS.items()]
    p["hybrid"] = [("v_hybrid", ["--methods", "DINOguidedSIFT", "SIFTdinoFilter", "SIFT",
                                 "--dino-ransac-threshold", "8"])]
    p["illum"] = [("i_identity_artifact", ["--categories", "i", "--methods", "DINOv3", "SIFT"]),
                  ("i_shift7", ["--categories", "i", "--methods", "DINOv3", "SIFT",
                                "--target-perturb", "shift", "--perturb-px", "7"]),
                  ("i_homography40", ["--categories", "i", "--methods", "DINOv3", "SIFT",
                                      "--target-perturb", "homography", "--perturb-px", "40"])]
    p["learned"] = [("v_learned", ["--methods", "LoFTR", "LightGlue", "SIFT"])]
    p["core"] = p["sift"] + p["ransac"] + p["resolution"] + p["refine"] + p["hybrid"] + p["illum"]
    p["all"] = p["core"] + p["models"] + p["learned"]
    return p


def collect(output_root, names):
    rows = []
    for name in names:
        path = output_root / name / "summary.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for key, s in report["summary"].items():
            q = report.get("match_quality", {}).get(key, {})
            o = report.get("original_resolution", {}).get(key, {})
            rows.append({"config": name, "method": key, "size": report.get("size"),
                         "model": (report.get("model") or "").split("/")[-1],
                         "dino_ransac": report.get("dino_ransac_threshold_model_px"),
                         "refine": report.get("refine"), "perturb": report.get("target_perturb"),
                         "pairs": s["pairs"], "median_err_canvas_px": s["median_error_registered_px"],
                         "success5": s["success_at_5px"], "success10": s["success_at_10px"],
                         "precision5": q.get("micro_precision_5px"), "precision16": q.get("micro_precision_16px"),
                         "hacc3_orig": o.get("homography_accuracy_3px"),
                         "mma3_orig": (o.get("mma_micro") or {}).get("3")})
    return rows


def fmt(v, pct=False):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    return f"{100 * v:.1f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(presets()), default="core")
    parser.add_argument("--output", type=Path, default=Path("results/sweep"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run finished configurations")
    args = parser.parse_args()
    configs = presets()[args.preset]
    script = Path(__file__).with_name("hpatches_spatial_eval.py")
    for name, extra in configs:
        out = args.output / name
        if (out / "summary.json").is_file() and not args.force:
            print(f"skip {name} (done)")
            continue
        command = [args.python, str(script), "--data", str(args.data), "--output", str(out),
                   "--visualize", str(args.visualize), *extra]
        if "--categories" not in extra:
            command += ["--categories", "v"]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    rows = collect(args.output, [n for n, _ in presets()["all"]])
    if not rows:
        print("No finished configurations yet.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sweep_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = ["config", "method", "size", "dino_ransac", "refine", "perturb", "median_err_canvas_px",
              "success5", "success10", "precision5", "precision16", "hacc3_orig", "mma3_orig"]
    pct = {"success5", "success10", "precision5", "precision16", "hacc3_orig", "mma3_orig"}
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h], h in pct) for h in header) + " |")
    (args.output / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
