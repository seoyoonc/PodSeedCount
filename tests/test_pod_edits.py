"""Seed split, cut moves, ignored / merged / split pods, count history, reviewed images as ground truth."""
import copy
import shutil

import numpy as np
import pytest

from conftest import DATA, ROOT
from seedcounter.config import load_config
from seedcounter.edits import apply_edits
from seedcounter.summary import image_summary, pod_summary


def _img(gt_results):
    return copy.deepcopy(gt_results[0][2])


def _count(img, cfg, k):
    return pod_summary(img.pod(k), cfg, img.seed_radius_px)


def test_split_seed_adds_one_and_restore_undoes(gt_results, cfg):
    img = _img(gt_results)
    pod = img.pods[0]
    s = max(pod.active_seeds(), key=lambda x: x.area_px)
    n0 = _count(img, cfg, pod.index)["seed_count"]
    x0, y0, x1, y1 = *np.min(s.contour, axis=0), *np.max(s.contour, axis=0)
    cx = (x0 + x1) / 2
    apply_edits(img, [{"op": "split", "seed": s.id, "line": [[cx, y0 - 5], [cx, y1 + 5]]}])
    r = _count(img, cfg, pod.index)
    assert r["seed_count"] == n0 + 1 and r["auto_count"] == n0 and r["change"] == 1
    parts = [x for x in pod.seeds if x.features.get("split_from") == s.id]
    assert len(parts) == 2 and len({p.id for p in parts} | {x.id for x in img.pods[0].seeds}) == len(img.pods[0].seeds)
    apply_edits(img, [{"op": "restore", "seed": s.id}])
    assert _count(img, cfg, pod.index)["seed_count"] == n0
    with pytest.raises(ValueError):  # line misses the seed
        apply_edits(img, [{"op": "split", "seed": s.id, "line": [[x1 + 50, y0], [x1 + 50, y1]]}])


def test_moving_cuts_changes_what_is_counted(gt_results, cfg):
    img = _img(gt_results)
    pod = img.pods[0]
    n0 = _count(img, cfg, pod.index)["seed_count"]
    xs = sorted(s.centroid[0] for s in pod.active_seeds())
    apply_edits(img, [{"op": "cut", "pod": pod.index, "which": "bud", "x": (xs[2] + xs[3]) / 2}])
    assert _count(img, cfg, pod.index)["seed_count"] == n0 - 3
    with pytest.raises(ValueError):
        apply_edits(img, [{"op": "cut", "pod": pod.index, "which": "stem", "x": xs[0]}])


def test_ignore_merge_and_split_pods(gt_results, cfg):
    img = _img(gt_results)
    total = image_summary(img, cfg)["seeds"]
    a, b = img.pods[0], img.pods[1]
    na, nb = len(a.active_seeds()), len(b.active_seeds())
    apply_edits(img, [{"op": "ignore_pod", "pod": b.index}])
    s = image_summary(img, cfg)
    assert s["seeds"] == total - nb and s["pods"] == len(img.pods) - 1 and "ignored" in s["note"]
    apply_edits(img, [{"op": "restore_pod", "pod": b.index}, {"op": "merge_pods", "pods": [a.index, b.index]}])
    merged = img.pod(a.index)
    assert len(merged.active_seeds()) == na + nb and image_summary(img, cfg)["seeds"] == total
    # split the merged pod again between the two rows of pods
    ys = [p[1] for p in merged.outline]
    mid = (np.mean([s.centroid[1] for s in merged.seeds if s.pod == a.index][:na]) +
           np.mean([s.centroid[1] for s in merged.seeds][na:])) / 2
    xmin, xmax = min(p[0] for p in merged.outline), max(p[0] for p in merged.outline)
    apply_edits(img, [{"op": "split_pod", "pod": merged.index, "line": [[xmin - 10, mid], [xmax + 10, mid]]}])
    assert len(img.pods) == len(gt_results[0][2].pods)
    assert image_summary(img, cfg)["seeds"] == total


def test_reviewed_images_become_ground_truth(tmp_path):
    from seedcounter.calibrate import GTSet, calibrate
    from seedcounter.review.session import ReviewSession, reviewed_ground_truth
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp)
    s = ReviewSession(inp, tmp_path / "out", load_config())
    s.load_all()
    key = s.keys[0]
    seed = s.current[key].pods[0].seeds[0]
    s.apply(key, {"op": "reject", "seed": seed.id})
    assert reviewed_ground_truth(tmp_path / "out")[0] == []          # not checked yet
    s.apply(key, {"op": "reviewed", "value": True})
    gts, warns = reviewed_ground_truth(tmp_path / "out")
    assert len(gts) == 1 and gts[0][1].max() == image_summary(s.current[key], s.cfg)["seeds"]
    gt = GTSet(None, s.cfg)
    gt.add(gts)
    res = calibrate(gt, s.cfg, sweep_grid=False)
    assert res["before"]["abs_error"] == 1 and "confidence.pod_error_tolerance" in res["changes"]
