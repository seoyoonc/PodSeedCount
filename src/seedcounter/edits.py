"""Manual corrections, applied on top of an automatic result.

A reviewer's actions are plain records (dicts) that can be saved as JSON and replayed.
``apply_edits`` changes the stored observations; counts, colours and review flags are then
recomputed by ``seedcounter.summary``. All coordinates are full-resolution image pixels.

Seeds
  {"op": "accept", "seed": "3-12"}                 confirm a seed (it becomes green)
  {"op": "reject", "seed": "3-12"}                 not a seed (kept for the record, not counted)
  {"op": "restore", "seed": "3-12"}                undo accept / reject / split
  {"op": "add", "pod": 3, "contour": [[x, y], ...]} a seed the program missed
  {"op": "delete", "seed": "3-61"}                 remove a manually added seed
  {"op": "split", "seed": "3-12", "line": [[x, y], [x, y]]}   one border holds two seeds: cut it
  {"op": "redraw", "seed": "3-12", "contour": [[x, y], ...]}  the border is wrong: replace it
       (only redrawn borders teach the border settings in calibration; "accept" only confirms detection)
Dark areas without a seed
  {"op": "dismiss", "pod": 3, "area": 0}  /  {"op": "undismiss", "pod": 3, "area": 0}
Pods
  {"op": "cut", "pod": 3, "which": "bud" | "stem", "x": 812.5}    move the bud or stem cut
  {"op": "ignore_pod", "pod": 3}  /  {"op": "restore_pod", "pod": 3}   not a pod (e.g. debris)
  {"op": "merge_pods", "pods": [3, 4]}                             one pod detected as two
  {"op": "split_pod", "pod": 3, "line": [[x, y], [x, y]]}         two touching pods detected as one
Image
  {"op": "reviewed", "value": true}                the whole image has been checked
"""
from __future__ import annotations

import cv2
import numpy as np

from .model import ImageResult, Pod, Seed


# ------------------------------------------------------------------ geometry helpers
def _contour_stats(pts: np.ndarray):
    m = cv2.moments(pts)
    area = abs(cv2.contourArea(pts))
    cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else tuple(pts.mean(0))
    if len(pts) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(pts)
        major, minor = max(a, b), min(a, b)
    else:
        major = minor = float(2 * np.sqrt(area / np.pi))
    return float(cx), float(cy), float(area), float(major), float(minor)


def _row(cx: float, cy: float, centerline) -> str:
    cl = np.asarray(centerline, float)
    cl_y = np.interp(cx, cl[:, 0], cl[:, 1]) if len(cl) else cy
    return "top" if cy < cl_y else "bottom"


def _manual_seed(img: ImageResult, pod: Pod, contour, features=None) -> Seed:
    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("a seed outline needs at least 3 points")
    cx, cy, area, major, minor = _contour_stats(pts)
    if area < 4:
        raise ValueError("the drawn seed is too small")
    return Seed(id=img.new_seed_id(pod.index), pod=pod.index, contour=pts.round(1).tolist(),
                centroid=[round(cx, 1), round(cy, 1)], area_px=round(area, 1), major_px=round(major, 1),
                minor_px=round(minor, 1), row=_row(cx, cy, pod.centerline), confidence=1.0,
                features=features or {}, source="manual", reviewed="accepted")


def _rasterize(polys, pad: int = 3):
    """Polygons -> (mask, x0, y0) in a local box around them."""
    allp = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in polys])
    x0, y0 = np.floor(allp.min(0)).astype(int) - pad
    x1, y1 = np.ceil(allp.max(0)).astype(int) + pad
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(np.asarray(p, np.float64).reshape(-1, 2) - [x0, y0]).astype(np.int32)], 1)
    return mask.astype(bool), x0, y0


def _side_masks(mask, x0, y0, line):
    (ax, ay), (bx, by) = np.asarray(line, float)
    if abs(ax - bx) + abs(ay - by) < 1:
        raise ValueError("the cut line is too short")
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    side = (bx - ax) * (yy + y0 - ay) - (by - ay) * (xx + x0 - ax) > 0
    return mask & side, mask & ~side


def _largest_contour(mask, x0, y0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) + [x0, y0]
    return c.round(1).tolist()


def _side_of(pt, line) -> bool:
    (ax, ay), (bx, by) = np.asarray(line, float)
    return (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax) > 0


def _outline_y_at(outline, x: float):
    """Top and bottom of a pod outline at column x."""
    pts = np.asarray(outline, float)
    ys = []
    for (xa, ya), (xb, yb) in zip(pts, np.roll(pts, -1, axis=0)):
        if (xa - x) * (xb - x) <= 0 and xa != xb:
            ys.append(ya + (x - xa) * (yb - ya) / (xb - xa))
    if len(ys) < 2:
        raise ValueError("that position is outside the pod")
    return min(ys), max(ys)


def _centerline_from_mask(mask, x0, y0, step: int = 8):
    cols = np.nonzero(mask.any(0))[0]
    pts = []
    for c in cols[::step]:
        ys = np.nonzero(mask[:, c])[0]
        pts.append([float(c + x0), float((ys.min() + ys.max()) / 2 + y0)])
    if len(pts) >= 5:
        ys = np.convolve(np.array([p[1] for p in pts]), np.ones(5) / 5, mode="same")
        ys[:2], ys[-2:] = [p[1] for p in pts[:2]], [p[1] for p in pts[-2:]]
        pts = [[p[0], round(float(y), 1)] for p, y in zip(pts, ys)]
    return pts


def _set_cut(pod: Pod, which: str, x: float) -> None:
    top, bot = _outline_y_at(pod.outline, x)
    line = [[round(x, 1), round(top, 1)], [round(x, 1), round(bot, 1)]]
    if which == "bud":
        pod.bud_cut = line
    elif which == "stem":
        pod.stem_cut = line
    else:
        raise ValueError("'which' must be 'bud' or 'stem'")


# ------------------------------------------------------------------ operations
GEOMETRY = ("contour", "centroid", "area_px", "major_px", "minor_px", "row")


def _redraw_seed(img: ImageResult, edit: dict) -> None:
    pod, seed = img.find_seed(edit["seed"])
    if seed.status != "active":
        raise ValueError("only counted seeds can be redrawn")
    pts = np.asarray(edit.get("contour") or [], np.float32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("draw the new border all the way around the seed")
    pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 0.5, True).reshape(-1, 2)  # drop redundant points
    if len(pts) < 3:
        raise ValueError("draw the new border all the way around the seed")
    cx, cy, area, major, minor = _contour_stats(pts)
    if area < 4:
        raise ValueError("the new border is too small")
    if "original" not in seed.features:  # keep the automatic border, so the redraw can be undone
        seed.features["original"] = {k: getattr(seed, k) for k in GEOMETRY}
    seed.contour = pts.round(1).tolist()
    seed.centroid = [round(cx, 1), round(cy, 1)]
    seed.area_px, seed.major_px, seed.minor_px = round(area, 1), round(major, 1), round(minor, 1)
    seed.row = _row(cx, cy, pod.centerline)
    seed.features["redrawn"] = True
    if seed.source == "auto":
        seed.reviewed = "accepted"


def _split_seed(img: ImageResult, edit: dict) -> None:
    pod, seed = img.find_seed(edit["seed"])
    if seed.status != "active":
        raise ValueError("only counted seeds can be split")
    mask, x0, y0 = _rasterize([seed.contour], pad=2)
    a, b = _side_masks(mask, x0, y0, edit["line"])
    total = mask.sum()
    if min(a.sum(), b.sum()) < 0.1 * total:
        raise ValueError("draw the line through the seed, so that both parts are reasonably large")
    parts = [_largest_contour(m, x0, y0) for m in (a, b)]
    if any(p is None or len(p) < 3 for p in parts):
        raise ValueError("the seed could not be split along that line")
    seed.status, seed.reviewed = "rejected", "split"
    for p in parts:
        pod.seeds.append(_manual_seed(img, pod, p, {"split_from": seed.id}))


def _merge_pods(img: ImageResult, edit: dict) -> None:
    idx = [int(i) for i in edit["pods"]]
    if len(idx) != 2 or idx[0] == idx[1]:
        raise ValueError("merge needs two different pods")
    a, b = img.pod(idx[0]), img.pod(idx[1])
    mask, x0, y0 = _rasterize([a.outline, b.outline])
    lab_n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    if lab_n > 2:  # not touching: join them with their convex hull
        hull = cv2.convexHull(np.concatenate([np.asarray(a.outline), np.asarray(b.outline)]).astype(np.float32))
        outline = hull.reshape(-1, 2).round(1).tolist()
    else:
        outline = _largest_contour(mask, x0, y0)
    first, last = (a, b) if a.bud_x <= b.bud_x else (b, a)
    for s in b.seeds:
        s.pod = a.index
    a.seeds += b.seeds
    a.uncovered += b.uncovered
    a.centerline = sorted(a.centerline + b.centerline, key=lambda p: p[0])
    a.outline = outline
    bud_x = first.bud_x
    stem_x = max(a.stem_x, b.stem_x)
    a.bud_visible, a.stem_visible = first.bud_visible, (a if a.stem_x >= b.stem_x else b).stem_visible
    a.body_width_px = max(a.body_width_px, b.body_width_px)
    a.warnings = sorted(set(a.warnings + b.warnings + [f"merged with pod {b.index} by reviewer"]))
    _set_cut(a, "bud", bud_x)
    _set_cut(a, "stem", stem_x)
    img.pods.remove(b)


def _split_pod(img: ImageResult, edit: dict) -> None:
    pod = img.pod(int(edit["pod"]))
    line = edit["line"]
    mask, x0, y0 = _rasterize([pod.outline])
    m1, m2 = _side_masks(mask, x0, y0, line)
    if min(m1.sum(), m2.sum()) < 0.05 * mask.sum():
        raise ValueError("draw the line all the way through the pod, where the two pods should be separated")
    parts = []
    for m in (m1, m2):
        outline = _largest_contour(m, x0, y0)
        if outline is None or len(outline) < 3:
            raise ValueError("the pod could not be split along that line")
        parts.append((m, outline))
    side_seeds = [[s for s in pod.seeds if _side_of(s.centroid, line) == flag] for flag in (True, False)]
    keep = 0 if len(side_seeds[0]) >= len(side_seeds[1]) else 1
    new_index = max(p.index for p in img.pods) + 1
    new_pod = None
    for k, (m, outline) in enumerate(parts):
        flag = (True, False)[k]
        target = pod if k == keep else Pod(index=new_index, outline=[], centerline=[], bud_cut=[], stem_cut=[],
                                             body_width_px=pod.body_width_px,
                                             bud_visible=pod.bud_visible, stem_visible=pod.stem_visible)
        target.outline = outline
        target.centerline = _centerline_from_mask(m, x0, y0)
        cols = np.nonzero(m.any(0))[0]
        xmin, xmax = cols.min() + x0, cols.max() + x0
        bud_x = min(max(pod.bud_x, xmin + 2), xmax - 2)
        stem_x = max(min(pod.stem_x, xmax - 2), bud_x + 1)
        moved = side_seeds[k]
        uncovered = [u for u in pod.uncovered if _side_of(np.mean(u["contour"], axis=0), line) == flag]
        if k != keep:
            new_pod = target
            for s in moved:
                s.pod = new_index
            target.seeds = moved
            target.uncovered = uncovered
            target.warnings = [f"split from pod {pod.index} by reviewer"]
        else:
            pod.seeds = moved
            pod.uncovered = uncovered
            pod.warnings = sorted(set(pod.warnings + ["split by reviewer"]))
        _set_cut(target, "bud", bud_x)
        _set_cut(target, "stem", stem_x)
    img.pods.append(new_pod)


def apply_edit(img: ImageResult, edit: dict) -> None:
    op = edit.get("op")
    if op in ("accept", "reject", "restore", "delete"):
        pod, seed = img.find_seed(edit["seed"])
        if op == "accept":
            seed.status, seed.reviewed = "active", "accepted"
        elif op == "reject":
            seed.status = "rejected"
        elif op == "restore":
            if seed.reviewed == "split":  # undo a split: remove its two parts
                pod.seeds = [s for s in pod.seeds if s.features.get("split_from") != seed.id]
            original = seed.features.pop("original", None)
            if original:  # undo a redraw
                for k in GEOMETRY:
                    setattr(seed, k, original[k])
                seed.features.pop("redrawn", None)
            if seed.source == "manual":
                seed.status = "active"
            else:
                seed.status, seed.reviewed = "active", None
        elif op == "delete":
            if seed.source != "manual":
                raise ValueError("only seeds you added can be deleted; reject automatic ones instead")
            pod.seeds.remove(seed)
    elif op == "add":
        pod = img.pod(int(edit["pod"]))
        if not edit.get("contour"):
            raise ValueError("'add' needs a contour")
        pod.seeds.append(_manual_seed(img, pod, edit["contour"]))
    elif op == "split":
        _split_seed(img, edit)
    elif op == "redraw":
        _redraw_seed(img, edit)
    elif op in ("dismiss", "undismiss"):
        pod = img.pod(int(edit["pod"]))
        i = int(edit["area"])
        if not 0 <= i < len(pod.uncovered):
            raise ValueError(f"pod {pod.index} has no dark area {i}")
        pod.uncovered[i]["dismissed"] = op == "dismiss"
    elif op == "cut":
        pod = img.pod(int(edit["pod"]))
        x = float(edit["x"])
        if edit["which"] == "bud" and x >= pod.stem_x - 5 or edit["which"] == "stem" and x <= pod.bud_x + 5:
            raise ValueError("the bud cut must stay left of the stem cut")
        _set_cut(pod, edit["which"], x)
    elif op in ("ignore_pod", "restore_pod"):
        img.pod(int(edit["pod"])).status = "ignored" if op == "ignore_pod" else "active"
    elif op == "merge_pods":
        _merge_pods(img, edit)
    elif op == "split_pod":
        _split_pod(img, edit)
    elif op == "reviewed":
        img.reviewed = bool(edit.get("value", True))
    else:
        raise ValueError(f"unknown edit operation: {op}")


def apply_edits(img: ImageResult, edits: list[dict]) -> ImageResult:
    for e in edits:
        apply_edit(img, e)
    return img
