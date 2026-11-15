import copy
import json

import pytest

from seedcounter.edits import apply_edits
from seedcounter.model import image_from_dict, to_dict
from seedcounter.summary import image_summary, pod_summary


def _first(gt_results):
    return copy.deepcopy(gt_results[0][2])


def test_roundtrip_through_json(gt_results, cfg):
    img = _first(gt_results)
    again = image_from_dict(json.loads(json.dumps(to_dict(img), default=float)))
    assert image_summary(again, cfg)["pod_rows"] == image_summary(img, cfg)["pod_rows"]


def test_edits_update_counts(gt_results, cfg):
    img = _first(gt_results)
    pod = img.pods[0]
    before = pod_summary(pod, cfg, img.seed_radius_px)["seed_count"]
    s = pod.seeds[0]
    apply_edits(img, [{"op": "reject", "seed": s.id}])
    assert pod_summary(pod, cfg, img.seed_radius_px)["seed_count"] == before - 1
    x, y = s.centroid
    apply_edits(img, [{"op": "restore", "seed": s.id},
                      {"op": "add", "pod": pod.index, "contour": [[x, y - 20], [x + 20, y], [x, y + 20], [x - 20, y]]}])
    after = pod_summary(pod, cfg, img.seed_radius_px)
    assert after["seed_count"] == before + 1
    added = pod.seeds[-1]
    assert added.source == "manual" and added.id not in {t.id for t in pod.seeds[:-1]}
    with pytest.raises(ValueError):
        apply_edits(img, [{"op": "delete", "seed": pod.seeds[0].id}])   # automatic seeds are rejected, not deleted
    apply_edits(img, [{"op": "delete", "seed": added.id}])
    assert pod_summary(pod, cfg, img.seed_radius_px)["seed_count"] == before


def test_accept_turns_seed_green_and_thresholds_recolour(gt_results, cfg):
    img = _first(gt_results)
    pod = img.pods[0]
    low = min(pod.seeds, key=lambda s: s.confidence)
    strict = copy.deepcopy(cfg)
    strict["confidence"]["review_threshold"] = 0.999
    strict["confidence"]["reject_threshold"] = 0.99
    assert pod_summary(pod, strict, img.seed_radius_px)["green"] < pod_summary(pod, cfg, img.seed_radius_px)["green"]
    apply_edits(img, [{"op": "accept", "seed": low.id}])
    from seedcounter.summary import seed_class
    assert seed_class(low, strict) == "green"
