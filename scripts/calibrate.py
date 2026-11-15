"""Tune settings against the ground truth in ground_truth/.

1. Sweeps the parameter grid in config (calibration.sweep) and picks the setting with the
   smallest per-pod count error (and best seed matching).
2. Fits the seed-confidence model to which detections matched an annotated seed.
3. Picks the pod error tolerance so that pods marked "no" are those whose count is right.

Ground truth: the annotations in ground_truth/ and, with --from-reviews, every image marked as
checked in the interactive review (your corrections become ground truth automatically).

Writes config/calibrated.yaml and a report in output/calibration/. With --apply, the new
values are also written into config/default.yaml (comments are kept; a backup is made).
"""
import argparse
import csv
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from seedcounter.calibrate import GTSet, apply_to_file, calibrate
from seedcounter.review.session import reviewed_ground_truth
from seedcounter.config import DEFAULT_CONFIG, load_config, save_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(ROOT / "ground_truth"), help="ground-truth folder")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("-o", "--output", default=str(ROOT / "output" / "calibration"))
    ap.add_argument("--from-reviews", nargs="?", const=str(ROOT / "output"), metavar="OUTPUT_DIR",
                    help="also use images marked as checked in the review (default folder: output)")
    ap.add_argument("--reviews-only", action="store_true", help="use only reviewed images, not ground_truth/")
    ap.add_argument("--no-sweep", action="store_true", help="keep the segmentation settings; only refit confidence")
    ap.add_argument("--apply", action="store_true", help="write the results into the settings file (backup kept)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    gt = GTSet(None if a.reviews_only else a.gt, cfg)
    if a.from_reviews or a.reviews_only:
        extra, warns = reviewed_ground_truth(a.from_reviews or ROOT / "output")
        gt.warnings += warns
        gt.add(extra)
        print(f"{len(extra)} checked images from the review.")
    for w in gt.warnings:
        print("Note:", w)
    if not gt.entries:
        print("Error: no usable ground truth. See MANUAL.md, sections 13 and 21.", file=sys.stderr)
        return 1
    print(f"{len(gt.entries)} ground-truth images.")
    bar = (lambda xs: tqdm(xs, desc="Sweep", unit="setting")) if tqdm else None
    res = calibrate(gt, cfg, not a.no_sweep, bar)
    if res["sweep"]:
        with open(out / "sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(res["sweep"][0]))
            w.writeheader()
            w.writerows(res["sweep"])
    if res["pods"]:
        with open(out / "pod_check.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(res["pods"][0]))
            w.writeheader()
            w.writerows(res["pods"])
    b, t = res["before"], res["after"]
    lines = ["# Calibration report", "",
             "Scores are measured on the same images the settings were tuned on, so they are optimistic.", "",
             f"- Images: {res['images']}, pods: {t['pods']}",
             f"- Per-pod count error: {b['abs_error']} before, {t['abs_error']} after ({(t['rel_error'] or 0) * 100:.1f} %)",
             f"- Seed F1: {(b['f1'] or 0):.3f} before, {(t['f1'] or 0):.3f} after "
             f"(recall {(t['recall'] or 0):.3f}, precision {(t['precision'] or 0):.3f})",
             (f"- Border overlap with {t['border_seeds']} redrawn seeds: {b['border_overlap']:.3f} before, "
              f"{t['border_overlap']:.3f} after (1 = identical borders)" if t.get("border_overlap") is not None
              else "- Border overlap: no redrawn seeds yet, so border shape was not used"), "",
             "| image | pod | detected | truth | error | pod confidence | needs review |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['image']} | {r['pod']} | {r['detected']} | {r['truth']} | {r['error']:+d} | {r['pod_confidence']:.2f} | {r['needs_review']} |"
              for r in res["pods"]]
    lines += ["", "Changed settings:", ""] + [f"- `{k}`: {v}" for k, v in res["changes"].items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_config(res["config"], ROOT / "config" / "calibrated.yaml")
    print("\n".join(lines[4:8]))
    print(f"Wrote config/calibrated.yaml and {out / 'report.md'}")
    if a.apply:
        backup = apply_to_file(a.config, res["changes"])
        load_config(a.config)  # check it still loads
        print(f"Applied to {a.config} (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
