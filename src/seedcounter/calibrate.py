"""Calibration against ground truth: parameter sweep, seed-confidence fit, pod-tolerance fit."""
from __future__ import annotations

import copy
import itertools
import logging
import re
from pathlib import Path

import numpy as np

from .config import get_dotted, set_dotted
from .confidence import fit_logistic, seed_confidence
from .evaluate import evaluate_image, totals
from .ground_truth import load_ground_truth, load_manifest
from .pipeline import process_image
from .summary import pod_summary

log = logging.getLogger(__name__)


class GTSet:
    """Ground-truth images loaded once and reused for every setting of the sweep."""

    def __init__(self, gt_dir, cfg):
        self.entries, self.warnings = load_manifest(gt_dir) if gt_dir else ([], [])
        self.labels = {}
        self.borders = {}   # person-drawn borders used to judge border fit
        sources = cfg.get("calibration", {}).get("border_sources", ["redrawn"])
        self.annotation_borders = "annotations" in sources
        for e in self.entries:
            try:
                _, lab, w = load_ground_truth(e, cfg)
                self.labels[e.image] = lab
                if self.annotation_borders:
                    self.borders[e.image] = lab
                self.warnings += [f"{e.image.name}: {x}" for x in w]
            except Exception as exc:
                self.warnings.append(f"{e.image.name}: annotation could not be used ({exc}); using counts only")
                self.labels[e.image] = None

    def add(self, entries_with_labels) -> None:
        """Extra ground truth, e.g. images checked in the review: [(GTEntry, labels), ...]."""
        def same(a, b):  # the same file, possibly copied to another folder
            return a.resolve() == b.resolve() or (a.name == b.name and a.stat().st_size == b.stat().st_size)
        for item in entries_with_labels:
            entry, labels = item[0], item[1]
            borders = item[2] if len(item) > 2 else None
            if any(same(e.image, entry.image) for e in self.entries):
                self.warnings.append(f"{entry.image.name}: also in ground_truth/; the reviewed version is used")
                self.entries = [e for e in self.entries if not same(e.image, entry.image)]
            self.entries.append(entry)
            self.labels[entry.image] = labels
            if borders is not None:
                self.borders[entry.image] = borders

    def run(self, cfg):
        results, evals = [], []
        for e in self.entries:
            img = process_image(e.image, cfg)
            if img.error:
                continue
            results.append(img)
            evals.append(evaluate_image(img, self.labels[e.image], e.pod_counts, cfg, self.borders.get(e.image)))
        return results, evals


def score_of(t: dict, cfg: dict) -> float:
    c = cfg["calibration"]
    score = c["count_weight"] * t["abs_error"] - c["f1_weight"] * (t["f1"] or 0.0)
    if t.get("border_overlap") is not None:  # only when person-redrawn borders exist
        score -= c.get("border_weight", 0.0) * t["border_overlap"]
    return score


def sweep(gt: GTSet, cfg: dict, progress=None) -> list[dict]:
    grid = cfg["calibration"]["sweep"]
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*grid.values()))
    for vals in (progress(combos) if progress else combos):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, vals):
            set_dotted(c, k, v)
        _, evals = gt.run(c)
        t = totals(evals)
        rows.append({**dict(zip(keys, vals)), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()},
                     "score": round(score_of(t, c), 3)})
    # best score; ties broken by closeness to the current defaults (stability)
    def dist(r):
        return sum(abs(grid[k].index(r[k]) - grid[k].index(get_dotted(cfg, k))) if get_dotted(cfg, k) in grid[k] else 0 for k in keys)
    rows.sort(key=lambda r: (r["score"], dist(r)))
    return rows


def fit_confidence(gt: GTSet, cfg: dict) -> tuple[dict, float, list[dict]]:
    """Fit the seed model on matched/unmatched detections, then the pod error tolerance."""
    results, evals = gt.run(cfg)
    feats, ys = [], []
    for img, ev in zip(results, evals):
        if not ev["seed_ok"]:
            continue
        for pod in img.pods:
            for s in pod.seeds:
                feats.append(s.features)
                ys.append(int(ev["seed_ok"].get(s.id, False)))
    model = cfg["confidence"]["seed_model"]
    if len(ys) >= 30 and 0 < sum(ys) < len(ys):
        model = fit_logistic(feats, ys)
    c = copy.deepcopy(cfg)
    c["confidence"]["seed_model"] = model
    # re-score seeds with the new model
    for img in results:
        for pod in img.pods:
            for s in pod.seeds:
                s.confidence = round(seed_confidence(s.features, model), 3)
    # pod tolerance: "no" should mean the count is within the accepted error
    good_rel = cfg["calibration"].get("good_pod_error", 0.05)
    pods = []
    for img, ev in zip(results, evals):
        for pod, row in zip(img.pods, ev["pods"]):
            if row["truth"] is not None:
                good = abs(row["error"]) <= max(1, good_rel * max(row["truth"], 1))
                pods.append((pod, img.seed_radius_px, good, row))
    best_tol, best_acc = cfg["confidence"]["pod_error_tolerance"], -1.0
    for tol in np.round(np.arange(0.05, 0.61, 0.01), 2):
        c["confidence"]["pod_error_tolerance"] = float(tol)
        flags = [pod_summary(p, c, r)["needs_review"] == "no" for p, r, _, _ in pods]
        goods = [g for *_, g, _ in pods]
        tp = sum(f and g for f, g in zip(flags, goods)); tn = sum((not f) and (not g) for f, g in zip(flags, goods))
        pos, neg = sum(goods), len(goods) - sum(goods)
        acc = 0.5 * ((tp / pos if pos else 1) + (tn / neg if neg else 1))
        if acc > best_acc + 1e-9:  # first (smallest, most cautious) tolerance with the best agreement
            best_tol, best_acc = float(tol), acc
    c["confidence"]["pod_error_tolerance"] = best_tol
    table = []
    for p, r, g, row in pods:
        ps = pod_summary(p, c, r)
        table.append({**row, "pod_confidence": ps["pod_confidence"], "needs_review": ps["needs_review"],
                      "within_tolerance": g})
    return model, best_tol, table


def calibrate(gt: GTSet, cfg: dict, sweep_grid: bool = True, progress=None) -> dict:
    """Full calibration. Returns before/after scores, the changed settings and the new config."""
    _, evals = gt.run(cfg)
    before = totals(evals)
    new = copy.deepcopy(cfg)
    changes, rows = {}, []
    if sweep_grid:
        rows = sweep(gt, new, progress)
        for k in new["calibration"]["sweep"]:
            set_dotted(new, k, rows[0][k])
            if rows[0][k] != get_dotted(cfg, k):
                changes[k] = rows[0][k]
    model, tol, pods = fit_confidence(gt, new)
    new["confidence"]["seed_model"] = model
    new["confidence"]["pod_error_tolerance"] = tol
    for k, v in model.items():
        changes[f"confidence.seed_model.{k}"] = v
    changes["confidence.pod_error_tolerance"] = tol
    _, evals = gt.run(new)
    return {"before": before, "after": totals(evals), "changes": changes, "config": new,
            "pods": pods, "sweep": rows, "images": len(gt.entries), "warnings": gt.warnings}


def apply_to_file(path, changes: dict) -> Path:
    """Write changed settings into a YAML config file, keeping comments; returns the backup path."""
    import shutil
    path = Path(path)
    backup = path.with_suffix(".backup.yaml")
    shutil.copy(path, backup)
    text = path.read_text(encoding="utf-8")
    for k, v in changes.items():
        text = update_yaml_text(text, k, v)
    path.write_text(text, encoding="utf-8")
    return backup


# ------------------------------------------------------------------ YAML update keeping comments
def update_yaml_text(text: str, dotted: str, value) -> str:
    """Replace the value of a nested key in YAML text, keeping layout and comments."""
    parts = dotted.split(".")
    lines = text.splitlines(keepends=True)
    start, indent = 0, -1
    for depth, part in enumerate(parts):
        pat = re.compile(r"^(\s*)" + re.escape(part) + r":(.*)$")
        for i in range(start, len(lines)):
            m = pat.match(lines[i].rstrip("\n"))
            if m and len(m.group(1)) > indent:
                if depth == len(parts) - 1:
                    original = lines[i].rstrip("\n")
                    rest = m.group(2)
                    new = f"{m.group(1)}{part}: {value}"
                    if "#" in rest:  # keep the comment in its original column
                        col = original.index("#", len(m.group(1)) + len(part) + 1)
                        new = new.ljust(col) if len(new) < col - 1 else new + "  "
                        new += original[col:]
                    lines[i] = new + "\n"
                    return "".join(lines)
                start, indent = i + 1, len(m.group(1))
                break
        else:
            raise KeyError(dotted)
    return "".join(lines)
