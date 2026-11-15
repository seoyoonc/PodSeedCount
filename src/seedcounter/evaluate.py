"""Comparing results with ground truth: per-pod count error and seed-level matching."""
from __future__ import annotations

import cv2
import numpy as np

from .model import ImageResult
from .summary import pod_summary


def _pod_of_point(img: ImageResult, x: float, y: float):
    for pod in img.pods:
        if cv2.pointPolygonTest(np.array(pod.outline, np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0:
            return pod.index
    return None


def border_overlap(img: ImageResult, border_labels) -> dict:
    """How well detected borders fit person-drawn borders.

    For every reference border, the detected seed that overlaps it most is taken and the overlap
    ratio (intersection over union, 0..1) computed; a reference seed with no detection scores 0.
    """
    if border_labels is None or not border_labels.any():
        return {"n": 0, "iou_sum": 0.0}
    H, W = border_labels.shape
    det = np.zeros((H, W), np.int32)
    k = 0
    for pod in img.pods:
        for s in pod.active_seeds():
            k += 1
            cv2.fillPoly(det, [np.round(np.asarray(s.contour)).astype(np.int32)], k)
    det_area = np.bincount(det.ravel(), minlength=k + 1)
    from scipy import ndimage as ndi
    n, total = 0, 0.0
    for lab, sl in enumerate(ndi.find_objects(border_labels), start=1):
        if sl is None:
            continue
        ref = border_labels[sl] == lab
        hits = det[sl][ref]
        n += 1
        hits = hits[hits > 0]
        if hits.size == 0:
            continue
        best = np.bincount(hits).argmax()
        inter = int((hits == best).sum())
        total += inter / (ref.sum() + det_area[best] - inter)
    return {"n": n, "iou_sum": total}


def evaluate_image(img: ImageResult, gt_labels, pod_counts, cfg: dict, border_labels=None) -> dict:
    """Per pod: detected count, truth, error; per seed: matched to an annotated seed or not;
    border overlap with person-drawn borders (border_labels)."""
    warnings = []
    outline_counts = {p.index: 0 for p in img.pods}
    seed_ok: dict[str, bool] = {}
    if gt_labels is not None:
        from skimage.measure import regionprops
        for p in regionprops(gt_labels):
            cy, cx = p.centroid
            k = _pod_of_point(img, cx, cy)
            if k is not None:
                outline_counts[k] += 1
        used = set()
        for pod in img.pods:
            for s in pod.active_seeds():
                x, y = (int(round(v)) for v in s.centroid)
                inside = 0 <= y < gt_labels.shape[0] and 0 <= x < gt_labels.shape[1]
                lab = int(gt_labels[y, x]) if inside else 0
                seed_ok[s.id] = bool(lab) and lab not in used
                if lab:
                    used.add(lab)
    if pod_counts is not None and len(pod_counts) != len(img.pods):
        warnings.append(f"{img.image}: {len(pod_counts)} hand-counted pods but {len(img.pods)} pods detected; "
                        "using outline counts per pod")
        pod_counts = None
    rows = []
    for i, pod in enumerate(img.pods):
        summ = pod_summary(pod, cfg, img.seed_radius_px, img.reviewed)
        truth = pod_counts[i] if pod_counts is not None else (outline_counts[pod.index] if gt_labels is not None else None)
        ids = [s.id for s in pod.active_seeds()]
        tp = sum(seed_ok.get(sid, False) for sid in ids)
        rows.append({
            "image": img.image, "pod": pod.index, "detected": summ["seed_count"], "truth": truth,
            "error": None if truth is None else summ["seed_count"] - truth,
            "outline_count": outline_counts[pod.index] if gt_labels is not None else None,
            "matched": tp if gt_labels is not None else None,
            "pod_confidence": summ["pod_confidence"], "needs_review": summ["needs_review"],
        })
    return {"pods": rows, "seed_ok": seed_ok, "warnings": warnings, "border": border_overlap(img, border_labels)}


def totals(evals: list[dict]) -> dict:
    rows = [r for e in evals for r in e["pods"] if r["truth"] is not None]
    abs_err = sum(abs(r["error"]) for r in rows)
    matched = sum(r["matched"] or 0 for r in rows if r["matched"] is not None)
    det = sum(r["detected"] for r in rows if r["matched"] is not None)
    ann = sum(r["outline_count"] or 0 for r in rows if r["outline_count"] is not None)
    recall = matched / ann if ann else None
    precision = matched / det if det else None
    f1 = 2 * recall * precision / (recall + precision) if recall and precision else None
    truth = sum(r["truth"] for r in rows)
    bn = sum(e.get("border", {}).get("n", 0) for e in evals)
    bs = sum(e.get("border", {}).get("iou_sum", 0.0) for e in evals)
    return {"pods": len(rows), "abs_error": abs_err, "rel_error": abs_err / truth if truth else None,
            "recall": recall, "precision": precision, "f1": f1,
            "border_seeds": bn, "border_overlap": bs / bn if bn else None}
