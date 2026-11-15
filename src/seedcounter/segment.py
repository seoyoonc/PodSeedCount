"""Seed detection inside one pod: centre peaks, watershed borders, centerline split of merged seeds."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


@dataclass
class Detection:
    labels: np.ndarray        # int label image (bbox crop), 0 = no seed
    split: set                # labels created by splitting a merged object
    smooth: np.ndarray        # smoothed seedness used for borders
    local: np.ndarray         # local (band-pass) contrast
    foreground: np.ndarray    # candidate seed pixels


def detect(s: np.ndarray, zone: np.ndarray, center_y: np.ndarray, r: float, cfg: dict) -> Detection:
    """s: seedness (crop); zone: allowed area; center_y: centerline row per crop column; r: seed radius."""
    rows = np.arange(s.shape[0])[:, None]
    top_side = rows < center_y[None, :]
    sc = cfg["segmentation"]
    s1 = cv2.GaussianBlur(s, (0, 0), max(sc["peak_sigma"] * r, 0.5))
    s2 = cv2.GaussianBlur(s, (0, 0), max(sc["dog_sigma"] * r, 0.5))
    local = np.where(zone, s1 - s2, 0)
    min_dist = max(1, int(sc["min_distance"] * r))

    # 1. seed centres = local maxima of smoothed darkness with enough local contrast
    peaks = peak_local_max(np.where(zone, s1, 0), min_distance=min_dist, exclude_border=False)
    peaks = [p for p in peaks if zone[p[0], p[1]] and local[p[0], p[1]] > sc["peak_contrast"]
             and s1[p[0], p[1]] > sc["seed_threshold"]]
    markers = np.zeros(s.shape, np.int32)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    fg = (local > 0) & (s1 > sc["seed_threshold"]) & zone
    markers[~fg] = 0
    labels = watershed(-s1, markers, mask=fg)

    # 2. an object much larger than a seed that lies substantially on both sides of the
    #    centerline is two seeds (one per row): split it at the centerline
    area = np.pi * r * r
    split: set = set()
    nxt = int(labels.max()) + 1
    for p in regionprops(labels):
        if p.area < sc["split_area"] * area:
            continue
        region = labels == p.label
        frac_top = (region & top_side).sum() / p.area
        if min(frac_top, 1 - frac_top) < sc["split_side"]:
            continue
        parts = []
        for side in (top_side, ~top_side):
            half = region & side
            if half.sum() < sc["min_area"] * area:
                continue
            sh = np.where(half, s1, 0)
            # a real second seed has its own darkest point away from the centerline;
            # one seed lying across the line only peaks at the cut itself
            pk = [q for q in peak_local_max(sh, min_distance=min_dist, num_peaks=3, exclude_border=False)
                  if half[q[0], q[1]] and abs(q[0] - center_y[q[1]]) >= sc["split_offset"] * r]
            if not pk and p.area >= sc["split_area_large"] * area:
                # big enough for two seeds even without a second dark centre (faint seed
                # overlapping a darker one): use the darkest point of that side
                pk = [np.unravel_index(np.argmax(sh), sh.shape)]
            parts.append((half, pk))
        if len(parts) < 2 or any(not pk for _, pk in parts):
            continue
        if p.area < sc["split_area_large"] * area:
            # two seeds show a lighter gap between their centres; one elongated seed does not
            a = max(parts[0][1], key=lambda q: s1[q[0], q[1]])
            b = max(parts[1][1], key=lambda q: s1[q[0], q[1]])
            t = np.linspace(0, 1, 25)
            line = s1[np.round(a[0] + t * (b[0] - a[0])).astype(int), np.round(a[1] + t * (b[1] - a[1])).astype(int)]
            if 1 - line.min() / max(min(s1[a[0], a[1]], s1[b[0], b[1]]), 1e-6) < sc["split_dip"]:
                continue
        labels[region] = 0
        for half, pk in parts:
            mk = np.zeros(s.shape, np.int32)
            for q in pk:
                mk[q[0], q[1]] = nxt
                split.add(nxt)
                nxt += 1
            sub = watershed(-s1, mk, mask=half)
            labels[sub > 0] = sub[sub > 0]

    # 3. drop fragments that are too small to be a seed
    for p in regionprops(labels):
        if p.area < sc["min_area"] * area:
            labels[labels == p.label] = 0
            split.discard(p.label)
    return Detection(labels, split, s1, local, fg)
