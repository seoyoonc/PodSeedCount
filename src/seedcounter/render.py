"""Annotated images: seed borders coloured by confidence, pod counts coloured by review status."""
from __future__ import annotations

import cv2
import numpy as np

from .model import ImageResult
from .summary import image_summary, pod_summary, seed_class

RGB = {
    "green": (34, 160, 60), "orange": (240, 140, 20), "red": (215, 40, 40),
    "no": (34, 160, 60), "maybe": (240, 140, 20), "yes": (215, 40, 40), "error": (215, 40, 40),
    "centerline": (230, 30, 30), "cut": (30, 110, 230), "outline": (40, 40, 40), "uncovered": (200, 0, 200),
    "rejected": (150, 150, 150),
}


def _poly(pts, scale):
    return (np.asarray(pts, np.float64) * scale).round().astype(np.int32).reshape(-1, 1, 2)


def render(img: ImageResult, rgb: np.ndarray, cfg: dict) -> np.ndarray:
    rc = cfg["render"]
    scale = float(rc.get("scale", 1.0))
    canvas = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else rgb.copy()
    H, W = canvas.shape[:2]
    lw = max(1, int(round(rc["line_width"] * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, H / 1400)
    summ = image_summary(img, cfg)
    for pod in img.pods:
        if pod.status != "active":
            cv2.polylines(canvas, [_poly(pod.outline, scale)], True, RGB["rejected"], max(1, lw - 1), cv2.LINE_AA)
            x, y = (int(v * scale) for v in np.asarray(pod.outline).min(axis=0))
            cv2.putText(canvas, "ignored", (x, max(20, y - 8)), font, fs * 0.8, RGB["rejected"], max(1, lw - 1), cv2.LINE_AA)
            continue
        ps = pod_summary(pod, cfg, img.seed_radius_px, img.reviewed)
        counted = {id(s) for s in pod.active_seeds()}
        cv2.polylines(canvas, [_poly(pod.outline, scale)], True, RGB["outline"], max(1, lw - 1), cv2.LINE_AA)
        cv2.polylines(canvas, [_poly(pod.centerline, scale)], False, RGB["centerline"], max(1, lw - 1), cv2.LINE_AA)
        cv2.polylines(canvas, [_poly(pod.bud_cut, scale)], False, RGB["cut"], max(1, lw - 1), cv2.LINE_AA)
        cv2.polylines(canvas, [_poly(pod.stem_cut, scale)], False, RGB["cut"], lw * 2, cv2.LINE_AA)
        for u in pod.uncovered:
            if u.get("dismissed"):
                continue
            cv2.polylines(canvas, [_poly(u["contour"], scale)], True, RGB["uncovered"], max(1, lw - 1), cv2.LINE_AA)
        for s in pod.seeds:
            if s.reviewed == "split":
                continue  # replaced by its two parts
            on = id(s) in counted
            colour = RGB[seed_class(s, cfg)] if on else RGB["rejected"]
            cv2.polylines(canvas, [_poly(s.contour, scale)], True, colour, lw if on else 1, cv2.LINE_AA)
        # pod count to the left of the bud
        x = int(min(p[0] for p in pod.outline) * scale)
        y = int(np.mean([p[1] for p in pod.bud_cut]) * scale)
        text = str(ps["seed_count"])
        (tw, th), _ = cv2.getTextSize(text, font, fs * 1.6, max(2, lw))
        tx = max(4, x - tw - int(20 * fs))
        if tx + tw > x - 4:  # no room left of the pod: write above its left end
            tx, y = max(4, x), int(min(p[1] for p in pod.outline) * scale) - int(12 * fs)
        cv2.rectangle(canvas, (tx - 6, y - th - 8), (tx + tw + 6, y + 8), (255, 255, 255), -1)
        cv2.putText(canvas, text, (tx, y), font, fs * 1.6, RGB[ps["needs_review"]], max(2, lw), cv2.LINE_AA)
    # header band
    band = int(70 * fs)
    head = np.full((band, W, 3), 255, np.uint8)
    status = summ["needs_review"]
    line = f"{img.image}   {summ['seeds']} seeds in {summ['pods']} pods   needs review: {status}" + ("   (checked by reviewer)" if img.reviewed else "")
    cv2.putText(head, line, (int(16 * fs), int(band * 0.62)), font, fs * 0.9, RGB["outline"], max(1, lw - 1), cv2.LINE_AA)
    legend = [("confident", RGB["green"]), ("check", RGB["orange"]), ("likely wrong", RGB["red"]),
              ("dark area without seed", RGB["uncovered"])]
    lx = W - int(16 * fs)
    for name, col in reversed(legend):
        (tw, _), _ = cv2.getTextSize(name, font, fs * 0.6, 1)
        lx -= tw + int(40 * fs)
        cv2.rectangle(head, (lx, int(band * 0.35)), (lx + int(22 * fs), int(band * 0.65)), col, max(1, lw - 1))
        cv2.putText(head, name, (lx + int(28 * fs), int(band * 0.62)), font, fs * 0.6, RGB["outline"], 1, cv2.LINE_AA)
    return np.vstack([head, canvas])
