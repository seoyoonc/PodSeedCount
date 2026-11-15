"""Derived results: seed colour classes, pod counts, pod confidence, review flags.

Everything here is recomputed from stored observations and the current thresholds, so it
stays correct after manual edits or threshold changes (no stored state to go stale).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .model import ImageResult, Pod, Seed

REVIEW_RANK = {"no": 0, "maybe": 1, "yes": 2}


def seed_class(seed: Seed, cfg: dict) -> str:
    if seed.source == "manual" or seed.reviewed == "accepted":
        return "green"  # confirmed by a person
    c = cfg["confidence"]
    if seed.confidence >= c["review_threshold"]:
        return "green"
    if seed.confidence >= c["reject_threshold"]:
        return "orange"
    return "red"


def review_flag(conf: float, cfg: dict) -> str:
    c = cfg["confidence"]
    if conf >= c["review_threshold"]:
        return "no"
    if conf >= c["reject_threshold"]:
        return "maybe"
    return "yes"


def _uncovered_seeds(pod: Pod, seed_area: float) -> float:
    """Seed-sized dark areas not covered by any active seed, in seeds' worth."""
    total = 0.0
    seeds = [np.array(s.contour, np.float32).reshape(-1, 1, 2) for s in pod.active_seeds() if len(s.contour) >= 3]
    for region in pod.uncovered:
        if region.get("dismissed"):
            continue  # a reviewer marked it as not a seed
        pts = np.array(region["contour"], np.float32)
        cx, cy = pts.mean(axis=0)
        if any(cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0 for c in seeds):
            continue  # a seed was added there
        total += region["area_px"] / seed_area
    return total


def edit_counts(pod: Pod) -> dict:
    return {"added": sum(s.source == "manual" and "split_from" not in s.features for s in pod.seeds),
            "split": sum(s.reviewed == "split" for s in pod.seeds),
            "rejected": sum(s.status == "rejected" and s.reviewed != "split" for s in pod.seeds),
            "accepted": sum(s.source == "auto" and s.reviewed == "accepted" and s.status == "active"
                            and not s.features.get("redrawn") for s in pod.seeds),
            "redrawn": sum(bool(s.features.get("redrawn")) and s.status == "active" for s in pod.seeds),
            "dismissed": sum(bool(u.get("dismissed")) for u in pod.uncovered)}


def pod_summary(pod: Pod, cfg: dict, seed_radius_px: float, reviewed: bool = False) -> dict:
    cc = cfg["confidence"]
    active = pod.active_seeds()
    classes = [seed_class(s, cfg) for s in active]
    n = len(active)
    counts = {k: classes.count(k) for k in ("green", "orange", "red")}
    unconfirmed = [s for s in active if s.source == "auto" and s.reviewed is None]
    expected_wrong = sum(1.0 - s.confidence for s in unconfirmed)
    uncovered = _uncovered_seeds(pod, math.pi * seed_radius_px ** 2) * cc["uncovered_weight"]
    ends = cc["missing_end_penalty"] * ((not pod.bud_visible) + (not pod.stem_visible))
    est_error = expected_wrong + uncovered + ends
    rel = est_error / max(n, 1)
    conf = float(np.clip(1.0 - rel / cc["pod_error_tolerance"], 0.0, 1.0))
    if pod.warnings and any("tilted" in w or "touching" in w for w in pod.warnings):
        conf = min(conf, cc["reject_threshold"])
    flag = review_flag(conf, cfg)
    reasons = []
    if reviewed:
        flag = "no"
        reasons.append("checked by reviewer")
    elif flag != "no":
        if counts["red"]:
            reasons.append(f"{counts['red']} red seed{'s' if counts['red'] > 1 else ''}")
        if counts["orange"]:
            reasons.append(f"{counts['orange']} orange seed{'s' if counts['orange'] > 1 else ''}")
        if uncovered >= 0.5:
            reasons.append(f"dark area worth about {uncovered:.0f} seed{'s' if round(uncovered) != 1 else ''} has no seed border")
        if not pod.bud_visible:
            reasons.append("bud end outside the image")
        if not pod.stem_visible:
            reasons.append("stem end outside the image")
        if n == 0:
            reasons.append("no seeds found")
    if not reviewed:
        reasons += [w for w in pod.warnings if w not in reasons and "outside the image" not in w and "touches" not in w]
    ec = edit_counts(pod)
    done = [f"{v} {k}" for k, v in ec.items() if v]
    if any("by reviewer" in w for w in pod.warnings) and reviewed:
        done += [w for w in pod.warnings if "by reviewer" in w]
    if done:
        reasons.append("edits: " + ", ".join(done))
    auto = sum(sd.source == "auto" for sd in pod.seeds)
    return {
        "pod": pod.index,
        "seed_count": n,
        "auto_count": auto,
        "change": n - auto,
        "top_row": sum(s.row == "top" for s in active),
        "bottom_row": sum(s.row == "bottom" for s in active),
        **counts,
        "pod_confidence": round(conf, 2),
        "needs_review": flag,
        "reason": "; ".join(reasons),
        "reviewed": "yes" if reviewed else "no",
    }


def image_summary(img: ImageResult, cfg: dict) -> dict:
    if img.error:
        return {"image": img.image, "pods": 0, "seeds": 0, "auto_seeds": 0, "change": 0, "abs_change": 0, "green": 0, "orange": 0, "red": 0,
                "pods_no": 0, "pods_maybe": 0, "pods_yes": 0, "needs_review": "error", "reviewed": "no",
                "annotated_image": "", "pod_rows": [], "note": img.error}
    rows = [pod_summary(p, cfg, img.seed_radius_px, img.reviewed) for p in img.active_pods()]
    status = max((r["needs_review"] for r in rows), key=REVIEW_RANK.get, default="yes")
    if img.reviewed:
        status = "no"
    note = "; ".join(img.warnings) if img.warnings else ""
    ignored = [p.index for p in img.pods if p.status != "active"]
    if ignored:
        note = "; ".join(x for x in [note, f"pod{'s' if len(ignored) > 1 else ''} {', '.join(map(str, ignored))} ignored by reviewer (not a pod)"] if x)
    if not rows:
        note = note or "no pods found"
    return {
        "image": img.image,
        "pods": len(rows),
        "seeds": sum(r["seed_count"] for r in rows),
        "auto_seeds": sum(r["auto_count"] for r in rows),
        "change": sum(r["change"] for r in rows),
        "abs_change": sum(abs(r["change"]) for r in rows),
        "green": sum(r["green"] for r in rows),
        "orange": sum(r["orange"] for r in rows),
        "red": sum(r["red"] for r in rows),
        "pods_no": sum(r["needs_review"] == "no" for r in rows),
        "pods_maybe": sum(r["needs_review"] == "maybe" for r in rows),
        "pods_yes": sum(r["needs_review"] == "yes" for r in rows),
        "needs_review": status,
        "reviewed": "yes" if img.reviewed else "no",
        "annotated_image": img.annotated_image or "",
        "pod_rows": rows,
        "note": note,
    }
