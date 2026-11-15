"""Seed-vs-pod contrast: relative darkness of the green channel against the local pod tone."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def seedness(rgb_crop: np.ndarray, pod_mask: np.ndarray, seed_radius: float, cfg: dict) -> np.ndarray:
    """0..1 map, 0 = pod background tone, higher = darker than the surrounding pod.

    Seeds absorb green light most strongly, so the green channel gives the best seed/pod
    separation on both the dense dark and the pale example images. The pod background is
    estimated with a grey closing larger than a seed (removes the seeds), so the map is
    normalised for pale and dark pods alike.
    """
    cc = cfg["contrast"]
    green = rgb_crop[..., 1].astype(np.float32)
    if cc["pre_blur"] > 0:
        green = cv2.GaussianBlur(green, (0, 0), cc["pre_blur"])
    filled = np.where(pod_mask, green, green[pod_mask].max())
    size = int(cc["background_size"] * seed_radius) | 1
    background = ndi.grey_closing(filled, size=(size, size))
    background = cv2.GaussianBlur(background, (0, 0), max(seed_radius, 1.0))
    s = np.clip((background - green) / np.maximum(background, 1.0), 0, 1)
    return np.where(pod_mask, s, 0).astype(np.float32)
