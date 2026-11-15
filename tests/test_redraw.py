"""Redrawing borders; only redrawn borders teach border shape in calibration."""
import copy
import shutil

import numpy as np
import pytest

from conftest import DATA, ROOT
from seedcounter.config import load_config
from seedcounter.edits import apply_edits
from seedcounter.evaluate import border_overlap
from seedcounter.summary import pod_summary, seed_class


def _circle(cx, cy, r, n=40):
    return [[cx + r * np.cos(a), cy + r * np.sin(a)] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def test_redraw_replaces_border_and_restore_brings_it_back(gt_results, cfg):
    img = copy.deepcopy(gt_results[0][2])
    pod = img.pods[0]
    s = pod.active_seeds()[3]
    before = copy.deepcopy(s)
    n0 = pod_summary(pod, cfg, img.seed_radius_px)["seed_count"]
    cx, cy = s.centroid
    apply_edits(img, [{"op": "redraw", "seed": s.id, "contour": _circle(cx, cy, 20)}])
    assert s.features["redrawn"] and s.reviewed == "accepted" and seed_class(s, cfg) == "green"
    assert abs(s.area_px - np.pi * 400) < 40 and s.contour != before.contour
    r = pod_summary(pod, cfg, img.seed_radius_px)
    assert r["seed_count"] == n0 and "1 redrawn" in r["reason"] or r["needs_review"] == "no"
    apply_edits(img, [{"op": "redraw", "seed": s.id, "contour": _circle(cx, cy, 15)}])   # second redraw
    assert s.features["original"]["contour"] == before.contour                         # keeps the automatic one
    apply_edits(img, [{"op": "restore", "seed": s.id}])
    assert s.contour == before.contour and s.reviewed is None and "redrawn" not in s.features
    with pytest.raises(ValueError):
        apply_edits(img, [{"op": "redraw", "seed": s.id, "contour": [[0, 0], [1, 1]]}])


def test_border_overlap_measure(gt_results):
    img = copy.deepcopy(gt_results[0][2])
    s = img.pods[0].active_seeds()[0]
    H, W = img.height, img.width
    import cv2
    ref = np.zeros((H, W), np.int32)
    cv2.fillPoly(ref, [np.round(np.asarray(s.contour)).astype(np.int32)], 1)
    assert border_overlap(img, ref)["iou_sum"] > 0.95                 # identical border
    ref2 = np.zeros((H, W), np.int32)
    cx, cy = s.centroid
    cv2.fillPoly(ref2, [np.round(np.asarray(_circle(cx, cy, 8))).astype(np.int32)], 1)
    assert border_overlap(img, ref2)["iou_sum"] < 0.5                  # much smaller drawn border
    assert border_overlap(img, None) == {"n": 0, "iou_sum": 0.0}


def test_only_redrawn_borders_teach_shape(tmp_path):
    from seedcounter.calibrate import GTSet, calibrate
    from seedcounter.review.session import ReviewSession, reviewed_ground_truth
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp)
    s = ReviewSession(inp, tmp_path / "out", load_config())
    s.load_all()
    key = s.keys[0]
    seeds = s.current[key].pods[1].active_seeds()
    s.apply(key, {"op": "accept", "seed": seeds[0].id})                       # accept: detection only
    s.apply(key, {"op": "reviewed", "value": True})
    (_, labels, borders), = reviewed_ground_truth(tmp_path / "out")[0]
    assert borders is None and labels.max() > 100                              # no redrawn border yet
    cx, cy = seeds[1].centroid
    s.apply(key, {"op": "redraw", "seed": seeds[1].id, "contour": _circle(cx, cy, 22)})
    (_, labels, borders), = reviewed_ground_truth(tmp_path / "out")[0]
    assert borders.max() == 1                                                  # exactly the redrawn seed
    gt = GTSet(None, s.cfg)
    gt.add(reviewed_ground_truth(tmp_path / "out")[0])
    res = calibrate(gt, s.cfg, sweep_grid=False)
    assert res["before"]["border_seeds"] == 1 and 0 < res["before"]["border_overlap"] < 1
