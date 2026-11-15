"""Finding pods and their geometry: outline, centerline, bud and stem cuts, seed zone."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class PodGeom:
    index: int
    bbox: tuple            # (y0, y1, x0, x1) in work pixels, exclusive end
    mask: np.ndarray       # bool, bbox crop
    x0: int                # first / last column of the pod (work pixels, absolute)
    x1: int
    top: np.ndarray        # per column x0..x1: top outline y (absolute)
    bot: np.ndarray
    center: np.ndarray     # smoothed centerline y per column
    width: np.ndarray
    body_width: float
    bud_x: float           # bud cut (absolute x): nothing left of this is counted
    stem_x: float          # stem cut (absolute x)
    bud_visible: bool = True
    stem_visible: bool = True
    warnings: list = field(default_factory=list)

    def center_at(self, xs):
        return np.interp(xs, np.arange(self.x0, self.x1 + 1), self.center)

    def outline_at(self, xs):
        cols = np.arange(self.x0, self.x1 + 1)
        return np.interp(xs, cols, self.top), np.interp(xs, cols, self.bot)


def pod_mask(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 2).astype(np.float32)
    val = hsv[..., 2]
    background = val >= np.percentile(val, 60)
    bg_sat = float(np.median(sat[background])) if background.any() else 0.0
    mask = sat > bg_sat + cfg["pods"]["saturation_margin"]
    mask = ndi.binary_opening(mask, iterations=2)
    return ndi.binary_fill_holes(mask)


def _columns(mask: np.ndarray, y_off: int, x_off: int):
    ys, xs = np.nonzero(mask)
    x0, x1 = xs.min(), xs.max()
    n = x1 - x0 + 1
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    order = np.argsort(xs, kind="stable")
    xs_s, ys_s = xs[order], ys[order]
    idx = np.searchsorted(xs_s, np.arange(x0, x1 + 2))
    for i in range(n):
        col = ys_s[idx[i]:idx[i + 1]]
        if col.size:
            top[i], bot[i] = col.min(), col.max()
    good = ~np.isnan(top)
    cols = np.arange(n)
    top = np.interp(cols, cols[good], top[good])
    bot = np.interp(cols, cols[good], bot[good])
    return x0 + x_off, x1 + x_off, top + y_off, bot + y_off


def find_pods(rgb: np.ndarray, cfg: dict, seed_radius: float) -> tuple[list[PodGeom], list[str]]:
    """Pods in a (work-scale) RGB image, sorted top to bottom."""
    pc = cfg["pods"]
    warnings: list[str] = []
    mask = pod_mask(rgb, cfg)
    lab, n = ndi.label(mask)
    if n == 0:
        return [], ["no pods found"]
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if sizes[i] >= pc["min_area_frac"] * sizes.max()]
    slices = ndi.find_objects(lab)
    H, W = mask.shape
    geoms = []
    for k in keep:
        sl = slices[k - 1]
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, H)
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        m = lab[y0:y1, x0:x1] == k
        px0, px1, top, bot = _columns(m, y0, x0)
        width = ndi.median_filter(bot - top + 1, 5)
        nn = len(width)
        body = float(np.median(width[int(nn * 0.2):max(int(nn * 0.8), int(nn * 0.2) + 1)]))
        win = max(5, int(0.6 * body))
        center = ndi.uniform_filter1d((top + bot) / 2.0, win, mode="nearest")
        wide_l = np.nonzero(width >= pc["left_body_frac"] * body)[0]
        wide_r = np.nonzero(width >= pc["right_body_frac"] * body)[0]
        bud_x = max(px0 + wide_l[0], px0 + pc["bud_frac"] * body)
        stem_x = px0 + wide_r[-1]
        g = PodGeom(0, (y0, y1, x0, x1), m, px0, px1, top, bot, center, width, body, float(bud_x), float(stem_x))
        if px0 <= 1:
            g.bud_visible = False
            g.warnings.append("bud end touches the image edge")
        if px1 >= W - 2 and width[-1] >= pc["right_body_frac"] * body:
            g.stem_visible = False
            g.warnings.append("stem end is outside the image")
        tilt = np.degrees(np.arctan(np.polyfit(np.arange(nn), center, 1)[0]))
        if abs(tilt) > pc["max_tilt_deg"]:
            g.warnings.append(f"pod tilted {tilt:.0f} degrees; results may be unreliable")
        if (y1 - y0) > 2.2 * body:
            g.warnings.append("pod outline is unusually tall; pods may be touching")
        if stem_x - bud_x < 4 * seed_radius:
            g.warnings.append("pod too short to analyse")
            continue
        geoms.append(g)
    geoms.sort(key=lambda g: float(np.mean(g.center)))
    for i, g in enumerate(geoms, start=1):
        g.index = i
    if not geoms:
        warnings.append("no pods found")
    return geoms, warnings


def seed_zone(g: PodGeom, cfg: dict, seed_radius: float) -> np.ndarray:
    """Where seeds may lie (bbox crop): inside the pod wall, between the bud and stem cuts."""
    pc = cfg["pods"]
    y0, y1, x0, x1 = g.bbox
    xs = np.arange(x0, x1)[None, :]
    dist = ndi.distance_transform_edt(g.mask)
    margin = pc["end_margin"] * seed_radius
    return g.mask & (xs >= g.bud_x + margin) & (xs <= g.stem_x - margin) & (dist >= pc["rim_frac"] * g.body_width)
