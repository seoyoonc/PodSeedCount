"""Compare the program's counts with the ground truth.

  python scripts/evaluate.py             per-pod comparison for every ground-truth image
  python scripts/evaluate.py --check-gt  check new annotations: outline counts vs. hand counts,
                                         plus images showing which outlines were recognised
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from skimage.segmentation import find_boundaries

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from seedcounter.config import DEFAULT_CONFIG, load_config
from seedcounter.evaluate import evaluate_image, totals
from seedcounter.ground_truth import load_ground_truth, load_manifest
from seedcounter.io import save_png
from seedcounter.pipeline import process_image


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=str(ROOT / "ground_truth"))
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("-o", "--output", default=str(ROOT / "output" / "evaluation"))
    ap.add_argument("--check-gt", action="store_true", help="check the annotations instead of the program")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    entries, warnings = load_manifest(a.gt)
    for w in warnings:
        print("Note:", w)
    evals, rows = [], []
    for e in entries:
        try:
            raw, labels, w = load_ground_truth(e, cfg)
        except Exception as exc:
            print(f"Problem: {e.image.name}: {exc}")
            continue
        for x in w:
            print(f"Note: {e.image.name}: {x}")
        img = process_image(e.image, cfg)
        if img.error:
            print(f"Problem: {e.image.name}: {img.error}")
            continue
        ev = evaluate_image(img, labels, e.pod_counts, cfg)
        for x in ev["warnings"]:
            print("Note:", x)
        evals.append(ev)
        rows += ev["pods"]
        if a.check_gt:
            hand = e.pod_counts or [None] * len(img.pods)
            print(f"\n{e.image.name} (annotation {e.annotation.name if e.annotation else 'none'})")
            for r, h in zip(ev["pods"], hand):
                flag = "" if h is None or r["outline_count"] == h else "   <- differs: check for open, missing or extra outlines"
                print(f"  pod {r['pod']}: outlines {r['outline_count']}, hand count {h}{flag}")
            if labels is not None:
                vis = raw.copy()
                vis[find_boundaries(labels)] = (255, 0, 255)
                save_png(out / f"{e.image.stem}_outlines.png", vis)
    if not rows:
        print("No ground truth could be evaluated.", file=sys.stderr)
        return 1
    with open(out / "evaluation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not a.check_gt:
        print(f"\n{'image':32s} pod  detected  truth  error  needs_review")
        for r in rows:
            err = "" if r["error"] is None else f"{r['error']:+d}"
            print(f"{r['image'][:32]:32s} {r['pod']:>3}  {r['detected']:>8}  {str(r['truth']):>5}  {err:>5}  {r['needs_review']}")
    t = totals(evals)
    print(f"\nTotal per-pod count error {t['abs_error']} over {t['pods']} pods"
          + (f"; seed recall {t['recall']:.3f}, precision {t['precision']:.3f}" if t["recall"] else ""))
    print(f"Details: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
