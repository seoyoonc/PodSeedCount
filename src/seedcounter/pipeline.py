"""Processing one image: pods -> contrast -> seeds -> confidence -> result model."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

from .confidence import seed_confidence
from .contrast import seedness
from .io import load_image
from .model import ImageResult, Pod, Seed
from .pods import find_pods, geom_from_outline, seed_zone
from .segment import detect

log = logging.getLogger(__name__)


def _largest_contour(mask: np.ndarray):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2) if cs else None


def _to_full(pts, y0, x0, scale):
    p = (np.asarray(pts, float) + [x0, y0]) / scale
    return p.round(1).tolist()


def _pod_observations(work: np.ndarray, g, cfg: dict, r: float, scale: float, debug: dict | None = None):
    """Seeds and evidence for one pod (geometry g): the Pod and its raw seed measurements."""
    area = math.pi * r * r
    y0, y1, x0, x1 = g.bbox
    s = seedness(work[y0:y1, x0:x1], g.mask, r, cfg)
    zone = seed_zone(g, cfg, r)
    xs = np.arange(x0, x1)
    top_side = (np.arange(y0, y1)[:, None]) < g.center_at(xs)[None, :]
    det = detect(s, zone, g.center_at(xs) - y0, r, cfg)
    if debug is not None:
        debug.setdefault("pods", []).append({"geom": g, "seedness": s, "zone": zone, "labels": det.labels})

    cols = np.arange(g.x0, g.x1 + 1, max(2, int(r / 3)))
    top, bot = g.outline_at(cols)
    outline = np.r_[np.c_[cols, top], np.c_[cols[::-1], bot[::-1]]]
    ccols = np.arange(int(g.bud_x), int(g.stem_x) + 1, max(2, int(r / 2)))
    pod = Pod(
        index=g.index,
        outline=(outline / scale).round(1).tolist(),
        centerline=(np.c_[ccols, g.center_at(ccols)] / scale).round(1).tolist(),
        bud_cut=(np.array([[g.bud_x, g.outline_at(g.bud_x)[0]], [g.bud_x, g.outline_at(g.bud_x)[1]]]) / scale).round(1).tolist(),
        stem_cut=(np.array([[g.stem_x, g.outline_at(g.stem_x)[0]], [g.stem_x, g.outline_at(g.stem_x)[1]]]) / scale).round(1).tolist(),
        body_width_px=round(g.body_width / scale, 1),
        bud_visible=g.bud_visible, stem_visible=g.stem_visible, warnings=list(g.warnings),
    )
    raw = []
    labels = det.labels
    for p in regionprops(labels, intensity_image=s):
        region = labels == p.label
        frac_top = (region & top_side).sum() / p.area
        cy, cx = p.centroid
        cx_abs = cx + x0
        end_dist = min(cx_abs - g.bud_x, g.stem_x - cx_abs) / r
        raw.append((pod, {
            "mask": region, "bbox": (y0, x0),
            "centroid": [(cx + x0) / scale, (cy + y0) / scale],
            "row": "top" if (cy + y0) < g.center_at(cx_abs) else "bottom",
            "area_px": p.area / scale ** 2,
            "major_px": p.axis_major_length / scale, "minor_px": p.axis_minor_length / scale,
            "features": {
                "darkness": round(float(p.intensity_mean), 4),
                "area_ratio": round(p.area / area, 3),
                "solidity": round(float(p.solidity), 3),
                "centerline_cross": round(float(min(frac_top, 1 - frac_top)), 3),
                "split": int(p.label in det.split),
                "end_distance": round(float(end_dist), 2),
            },
        }))
    # dark seed-like areas that no seed border covers (evidence of missed seeds)
    covered = ndi.binary_dilation(labels > 0, iterations=max(1, int(r / 5)))
    open_dark = det.foreground & ~covered
    lab_u, n_u = ndi.label(open_dark)
    for i in range(1, n_u + 1):
        m = lab_u == i
        if m.sum() >= 0.5 * area:
            c = _largest_contour(m)
            if c is not None:
                pod.uncovered.append({"contour": _to_full(c, y0, x0, scale), "area_px": round(m.sum() / scale ** 2, 1)})
    return pod, raw


def _attach_seeds(pods: list, raw: list, cfg: dict, scale: float, ref: float | None = None) -> None:
    """Turn raw measurements into Seeds with confidences (ref: typical seed darkness of the image)."""
    if ref is None:
        darks = [d["features"]["darkness"] for _, d in raw if not d["features"]["split"]] or [1.0]
        ref = float(np.median(darks)) or 1.0
    model = cfg["confidence"]["seed_model"]
    by_pod: dict[int, list] = {}
    for pod, d in raw:
        d["features"]["contrast"] = round(d["features"]["darkness"] / ref, 3)
        by_pod.setdefault(id(pod), []).append(d)
    for pod in pods:
        items = sorted(by_pod.get(id(pod), []), key=lambda d: d["centroid"][0])
        for n, d in enumerate(items, start=1):
            c = _largest_contour(d["mask"])
            if c is None:
                continue
            y0, x0 = d["bbox"]
            pod.seeds.append(Seed(
                id=f"{pod.index}-{n}", pod=pod.index, contour=_to_full(c, y0, x0, scale),
                centroid=[round(v, 1) for v in d["centroid"]], area_px=round(d["area_px"], 1),
                major_px=round(d["major_px"], 1), minor_px=round(d["minor_px"], 1), row=d["row"],
                confidence=round(seed_confidence(d["features"], model), 3), features=d["features"]))


def _work_image(rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    scale = cfg["processing"]["work_scale"]
    work = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else rgb
    return work, scale


def _analyse(rgb: np.ndarray, cfg: dict, radius_full: float, debug: dict | None = None):
    work, scale = _work_image(rgb, cfg)
    r = radius_full * scale
    geoms, warnings = find_pods(work, cfg, r)
    pods, raw = [], []          # raw: (pod, props dict) for the confidence pass
    for g in geoms:
        pod, items = _pod_observations(work, g, cfg, r, scale, debug)
        pods.append(pod)
        raw += items
    # confidence needs the image-wide typical seed darkness as reference
    _attach_seeds(pods, raw, cfg, scale)
    # seed-size estimate from clean, unsplit seeds
    clean = [d["area_px"] for _, d in raw if not d["features"]["split"] and d["features"]["solidity"] > 0.9]
    radius_est = float(np.sqrt(np.median(clean) / math.pi)) if len(clean) >= 5 else None
    return pods, warnings, radius_est


def analyse_drawn_pod(rgb: np.ndarray, cfg: dict, radius_full: float, outline_full, index: int = 0,
                      ref_darkness: float | None = None) -> Pod:
    """Analyse one pod whose outline a reviewer drew (full-resolution polygon), exactly like an
    automatically found pod: midline and cuts from the outline, then seeds inside it.
    ref_darkness: typical seed darkness of the rest of the image, so confidences stay comparable."""
    work, scale = _work_image(rgb, cfg)
    r = radius_full * scale
    pts = np.asarray(outline_full, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("trace the pod outline all the way round")
    g = geom_from_outline(pts * scale, work.shape[:2], cfg, r)
    g.index = index
    pod, raw = _pod_observations(work, g, cfg, r, scale)
    _attach_seeds([pod], raw, cfg, scale, ref_darkness)
    pod.outline = np.round(pts, 1).tolist()   # keep the reviewer's own outline
    return pod


def process_image(path: str | Path, cfg: dict, debug: dict | None = None) -> ImageResult:
    path = Path(path)
    try:
        rgb = load_image(path)
    except Exception as exc:  # unreadable file: report it, keep going with the rest
        log.error("Cannot read %s: %s", path.name, exc)
        return ImageResult(image=path.name, path=str(path), width=0, height=0, seed_radius_px=0, error=f"cannot read image: {exc}")
    pc = cfg["processing"]
    radius = float(pc["seed_radius_px"])
    try:
        pods, warnings, est = _analyse(rgb, cfg, radius, debug)
        # Detected borders run slightly larger than drawn outlines, so the estimate is only used
        # to correct a clearly different magnification, not to fine-tune.
        if pc.get("auto_radius") and est and abs(est / radius - 1) > pc["auto_radius_tolerance"]:
            log.info("%s: seed radius re-estimated as %.1f px (expected %.1f)", path.name, est, radius)
            warnings = warnings + [f"seed size differs from the default; re-analysed with radius {est:.0f} px"]
            radius = est
            if debug is not None:
                debug.clear()
            pods, w2, _ = _analyse(rgb, cfg, radius, debug)
    except Exception as exc:
        log.exception("Processing failed for %s", path.name)
        return ImageResult(image=path.name, path=str(path), width=rgb.shape[1], height=rgb.shape[0],
                           seed_radius_px=radius, error=f"processing failed: {exc}")
    return ImageResult(image=path.name, path=str(path), width=int(rgb.shape[1]), height=int(rgb.shape[0]),
                       seed_radius_px=round(radius, 2), pods=pods, warnings=warnings)
