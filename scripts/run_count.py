"""Count seeds in every TIFF of a folder.

Examples (from the seed_project folder):
  python scripts/run_count.py
  python scripts/run_count.py path/to/tifs -o path/to/output
  python scripts/run_count.py --review-threshold 0.8 --reject-threshold 0.5
  python scripts/run_count.py --set output.seeds_csv=true --set output.results_json=full
"""
import argparse
import logging
import sys

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from seedcounter.config import DEFAULT_CONFIG, load_config
from seedcounter.run import run_folder

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count seeds in pod images and write annotated images, tables and a report.")
    ap.add_argument("input", nargs="?", default=str(ROOT / "input_images"), help="folder with .tif images (default: input_images)")
    ap.add_argument("-o", "--output", default=str(ROOT / "output"), help="output folder (default: output)")
    ap.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="settings file (default: config/default.yaml)")
    ap.add_argument("--review-threshold", type=float, help="confidence at or above which seeds are green / pods 'no'")
    ap.add_argument("--reject-threshold", type=float, help="confidence below which seeds are red / pods 'yes'")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any setting, e.g. --set segmentation.min_area=0.25 (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    overrides = dict(kv.split("=", 1) for kv in a.set)
    if a.review_threshold is not None:
        overrides["confidence.review_threshold"] = a.review_threshold
    if a.reject_threshold is not None:
        overrides["confidence.reject_threshold"] = a.reject_threshold
    try:
        cfg = load_config(a.config, overrides)
        bar = (lambda xs: tqdm(xs, desc="Images", unit="img")) if tqdm else None
        run, sums = run_folder(a.input, a.output, cfg, a.config, bar)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    n_rev = sum(s["needs_review"] != "no" for s in sums)
    print(f"\n{len(sums)} images processed, {n_rev} need review.")
    for s in sums:
        counts = ", ".join(str(r["seed_count"]) for r in s["pod_rows"]) or s.get("note", "")
        print(f"  {s['image']}: {s['seeds']} seeds [{counts}]  needs review: {s['needs_review']}")
    print(f"Report: {a.output}/report.html")
    from seedcounter import __version__
    from seedcounter.settings_info import fingerprint
    print(f"Program version {__version__}, settings fingerprint {fingerprint(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
