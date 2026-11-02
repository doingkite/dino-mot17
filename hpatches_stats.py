"""Statistics and per-sequence analysis for hpatches_spatial_eval.py pairs.csv files.

No SciPy needed. Produces:
  * per-method summary with SEQUENCE-level bootstrap 95% CIs (pairs from one sequence are
    not independent, so we resample the 59 sequences, not the 295 pairs)
  * paired tests against a reference method: Wilcoxon signed-rank (normal approx., tie
    corrected) and exact sign test on per-pair median overlap error
  * error decomposition: near-miss (5 < e <= 16 px) vs gross (> 16 px) proposed matches
  * per-sequence table, worst sequences, by-target trend
Usage:
  python hpatches_stats.py results/hpatches_match_quality_v/pairs.csv --reference SIFT \
      --output results/hpatches_match_quality_v/stats
  python hpatches_stats.py A=results/a/pairs.csv B=results/b/pairs.csv   # merge runs, methods renamed A:DINOv3 ...
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FAIL_VALUE = 1e6  # failed registration ranks as worst in paired tests


def to_float(value):
    if value in (None, "", "None", "nan"):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def load_rows(specs):
    rows = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            label, path = "", spec
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if label:
                    row["method"] = f"{label}:{row['method']}"
                row["target"] = int(row["target"])
                for key, value in list(row.items()):
                    if key not in ("category", "sequence", "method", "registered"):
                        row[key] = to_float(value) if not isinstance(value, int) else value
                rows.append(row)
    if not rows:
        raise ValueError("No rows loaded")
    return rows


def pair_metrics(group):
    """Metrics computed on a list of rows (one method)."""
    errors = [r["median_overlap_error_px"] for r in group]
    finite = [e for e in errors if e is not None]
    out = {"pairs": len(group),
           "median_error_px": float(np.median(finite)) if finite else None,
           "success_5px": sum(e is not None and e <= 5 for e in errors) / len(group),
           "success_10px": sum(e is not None and e <= 10 for e in errors) / len(group)}
    total = sum(r.get("matches") or 0 for r in group)
    if total and group[0].get("correct_matches_5px") is not None:
        for t in (5, 10, 16):
            out[f"precision_{t}px"] = sum(r[f"correct_matches_{t}px"] for r in group) / total
    if "corner_error_orig_px" in group[0]:
        corners = [r["corner_error_orig_px"] for r in group]
        for t in (1, 3, 5, 10):
            out[f"homography_acc_{t}px_orig"] = sum(c is not None and c <= t for c in corners) / len(group)
        if total and group[0].get("correct_orig_3px") is not None:
            out["mma_3px_orig"] = sum(r["correct_orig_3px"] for r in group) / total
    return out


def bootstrap(group, iterations, seed):
    by_sequence = defaultdict(list)
    for row in group:
        by_sequence[row["sequence"]].append(row)
    names = sorted(by_sequence)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(iterations):
        chosen = rng.choice(len(names), len(names), replace=True)
        resampled = [row for i in chosen for row in by_sequence[names[i]]]
        for key, value in pair_metrics(resampled).items():
            if key != "pairs" and value is not None:
                samples[key].append(value)
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in samples.items()}


def normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon_signed_rank(differences):
    d = np.asarray([x for x in differences if x != 0], dtype=np.float64)
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "W_plus": None, "z": None, "p_two_sided": 1.0}
    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    ranks = np.empty(n)
    sorted_abs = abs_d[order]
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    z = (w_plus - mean) / math.sqrt(var) if var > 0 else 0.0
    return {"n_nonzero": n, "W_plus": w_plus, "z": z, "p_two_sided": min(1.0, 2 * normal_sf(abs(z)))}


def sign_test(wins_a, wins_b):
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    # exact two-sided binomial(n, 0.5); log-space for large n
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    top = max(log_terms)
    tail = math.exp(top) * sum(math.exp(t - top) for t in log_terms)
    return min(1.0, 2 * tail)


def paired_tests(rows, method, reference):
    index = defaultdict(dict)
    for row in rows:
        index[(row["category"], row["sequence"], row["target"])][row["method"]] = row["median_overlap_error_px"]
    diffs, wins_m, wins_r = [], 0, 0
    for methods in index.values():
        if method in methods and reference in methods:
            a = methods[method] if methods[method] is not None else FAIL_VALUE
            b = methods[reference] if methods[reference] is not None else FAIL_VALUE
            diffs.append(a - b)
            wins_m += a < b
            wins_r += b < a
    if not diffs:
        return None
    return {"pairs": len(diffs), f"{method}_better": int(wins_m), f"{reference}_better": int(wins_r),
            "median_difference_px": float(np.median(diffs)),
            "wilcoxon": wilcoxon_signed_rank(diffs),
            "sign_test_p_two_sided": sign_test(int(wins_m), int(wins_r)),
            "note": "difference = method - reference on per-pair median overlap error; failures = +1e6"}


def error_decomposition(group):
    total = sum(r.get("matches") or 0 for r in group)
    if not total or group[0].get("correct_matches_5px") is None:
        return None
    c5 = sum(r["correct_matches_5px"] for r in group)
    c16 = sum(r["correct_matches_16px"] for r in group)
    return {"proposed": int(total), "within_5px": c5 / total,
            "near_miss_5_to_16px": (c16 - c5) / total, "gross_over_16px_or_invisible": (total - c16) / total,
            "share_of_errors_that_are_near_miss": (c16 - c5) / (total - c5) if total > c5 else None}


def per_sequence(rows, methods):
    table = defaultdict(dict)
    for row in rows:
        table[row["sequence"]].setdefault(row["method"], []).append(row["median_overlap_error_px"])
    out = []
    for sequence in sorted(table):
        item = {"sequence": sequence}
        for m in methods:
            errs = table[sequence].get(m, [])
            finite = [e for e in errs if e is not None]
            item[f"{m}_median_px"] = float(np.median(finite)) if finite else None
            item[f"{m}_success5"] = sum(e is not None and e <= 5 for e in errs)
        out.append(item)
    return out


def fmt(value, pct=False):
    if value is None:
        return "–"
    return f"{100 * value:.1f}%" if pct else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", nargs="+", help="pairs.csv paths, optionally LABEL=path")
    parser.add_argument("--reference", default="SIFT")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--output", type=Path, default=Path("results/hpatches_stats"))
    args = parser.parse_args()
    rows = load_rows(args.pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["method"])].append(row)
    report = {"inputs": args.pairs, "methods": {}, "paired_tests": {}, "error_decomposition": {}, "by_target": {}}
    for (category, method), group in sorted(groups.items()):
        key = f"{category}/{method}"
        report["methods"][key] = {"point": pair_metrics(group),
                                  "ci95_sequence_bootstrap": bootstrap(group, args.bootstrap, args.seed),
                                  "sequences": len({r['sequence'] for r in group})}
        report["error_decomposition"][key] = error_decomposition(group)
        report["by_target"][key] = {str(t): pair_metrics([r for r in group if r["target"] == t])
                                    for t in sorted({r["target"] for r in group})}
        reference_key = next((m for (c, m) in groups if c == category and
                              (m == args.reference or m.endswith(":" + args.reference))), None)
        if reference_key and reference_key != method:
            report["paired_tests"][f"{key}_vs_{reference_key}"] = paired_tests(
                [r for r in rows if r["category"] == category], method, reference_key)
    methods = sorted({r["method"] for r in rows})
    sequences = per_sequence(rows, methods)
    report["worst_sequences"] = {
        m: sorted((s for s in sequences if s[f"{m}_median_px"] is not None),
                  key=lambda s: -s[f"{m}_median_px"])[:10] for m in methods}
    (args.output / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "per_sequence.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sequences[0]))
        writer.writeheader()
        writer.writerows(sequences)

    lines = ["# HPatches statistics", "", f"Inputs: {', '.join(args.pairs)}", "",
             "## Summary (95% CI: sequence-level bootstrap)", "",
             "| method | pairs | median err (canvas px) | success@5 | precision@5 | homography acc@3 (orig) |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, item in report["methods"].items():
        p, ci = item["point"], item["ci95_sequence_bootstrap"]

        def cell(name, pct):
            if p.get(name) is None:
                return "–"
            band = ci.get(name)
            return fmt(p[name], pct) + (f" [{fmt(band[0], pct)}, {fmt(band[1], pct)}]" if band else "")
        lines.append(f"| {key} | {p['pairs']} | {cell('median_error_px', False)} | {cell('success_5px', True)} | "
                     f"{cell('precision_5px', True)} | {cell('homography_acc_3px_orig', True)} |")
    lines += ["", "## Paired tests vs reference", ""]
    for key, test in report["paired_tests"].items():
        if test:
            w = test["wilcoxon"]
            lines.append(f"- **{key}**: {test['pairs']} pairs, wins {list(test.values())[1]} vs "
                         f"{list(test.values())[2]}, median diff {test['median_difference_px']:.2f}px, "
                         f"Wilcoxon z={w['z']:.2f} p={w['p_two_sided']:.2e}, sign test p={test['sign_test_p_two_sided']:.2e}")
    lines += ["", "## Where do wrong matches land? (all proposed matches)", "",
              "| method | ≤5px | 5–16px near miss | >16px / invisible | near-miss share of errors |",
              "|---|---:|---:|---:|---:|"]
    for key, d in report["error_decomposition"].items():
        if d:
            lines.append(f"| {key} | {fmt(d['within_5px'], True)} | {fmt(d['near_miss_5_to_16px'], True)} | "
                         f"{fmt(d['gross_over_16px_or_invisible'], True)} | {fmt(d['share_of_errors_that_are_near_miss'], True)} |")
    lines += ["", "## Worst sequences (median canvas error over targets 2–6)", ""]
    for m, worst in report["worst_sequences"].items():
        lines.append(f"- **{m}**: " + ", ".join(f"{s['sequence']} {s[f'{m}_median_px']:.1f}px" for s in worst[:6]))
    lines += ["", "## By target index (success@5)", ""]
    for key, targets in report["by_target"].items():
        lines.append(f"- {key}: " + ", ".join(f"{t}:{fmt(v['success_5px'], True)}" for t, v in targets.items()))
    (args.output / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
