"""Synthetic checks for pod outline detection (no example images needed): pale pods and pale bud ends are
found whole, uneven illumination and dust are not pods, and the earlier saturation method is still there."""
import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from seedcounter.config import load_config
from seedcounter.pods import find_pods, pod_mask

BG = (252, 250, 254)             # near-white with a slight blue-pink cast, like the real images


def _pale_pods(vignette: bool = False, dust: bool = False) -> np.ndarray:
    """Three pods whose left halves are very pale (saturation well below the old margin of 30)."""
    img = np.full((360, 1000, 3), BG, np.uint8)
    for y in (70, 180, 290):
        cv2.ellipse(img, (500, y), (420, 34), 0, 0, 360, (246, 243, 226), -1)   # very pale wall
        cv2.ellipse(img, (650, y), (260, 30), 0, 0, 360, (238, 228, 175), -1)   # yellower right half
        for x in range(520, 880, 50):                                           # a few seeds
            cv2.ellipse(img, (x, y - 12), (16, 11), 0, 0, 360, (200, 160, 70), -1)
    if vignette:
        yy, xx = np.mgrid[0:360, 0:1000]
        r = np.hypot((xx - 500) / 500, (yy - 180) / 180)
        img = np.clip(img.astype(np.float32) * (1 - 0.06 * r[..., None] ** 2), 0, 255).astype(np.uint8)
    if dust:
        for x, y in ((60, 20), (950, 340), (40, 330)):
            cv2.circle(img, (x, y), 3, (150, 150, 150), -1)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


@pytest.mark.nodata
def test_pale_pods_found_whole():
    cfg = load_config()
    m = pod_mask(_pale_pods(), cfg)
    lab, n = ndi.label(m)
    assert n == 3
    for y in (70, 180, 290):                       # the pale left end is part of the pod
        assert m[y, 110] and m[y, 300]


@pytest.mark.nodata
def test_saturation_method_loses_pale_halves():
    """The earlier method misses the pale halves; the setting still selects it."""
    cfg = load_config(overrides={"pods.method": "saturation"})
    m = pod_mask(_pale_pods(), cfg)
    assert not m[180, 200]


@pytest.mark.nodata
def test_vignetting_and_dust_are_not_pods():
    cfg = load_config()
    m = pod_mask(_pale_pods(vignette=True, dust=True), cfg)
    assert ndi.label(m)[1] == 3
    assert not m[10, 10] and not m[350, 990]        # darker corners stay background
    geoms, _ = find_pods(_pale_pods(vignette=True, dust=True), cfg, seed_radius=12)
    assert len(geoms) == 3
    assert all(g.x0 < 120 for g in geoms)            # each pod starts at its pale bud end
