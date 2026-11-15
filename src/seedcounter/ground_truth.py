"""Ground truth: manifest loading and extraction of hand-drawn seed outlines.

Annotations are full-frame copies of the raw image (any resolution) with closed white
outlines around each seed. Outlines are found as pixels that became whiter than the raw
image, then the inside of every closed loop is taken as one seed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from .io import load_image
from .pods import pod_mask

log = logging.getLogger(__name__)
ANNOTATION_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass
class GTEntry:
    image: Path
    annotation: Path | None
    pod_counts: list | None
    warnings: list = field(default_factory=list)


def load_manifest(gt_dir: str | Path) -> tuple[list[GTEntry], list[str]]:
    """Valid entries + warnings. A broken entry never stops the others."""
    gt_dir = Path(gt_dir)
    raw_dir, ann_dir = gt_dir / "raw", gt_dir / "annotations"
    warnings, entries = [], []
    manifest = gt_dir / "manifest.yaml"
    items = []
    if manifest.exists():
        try:
            items = yaml.safe_load(manifest.read_text(encoding="utf-8")) or []
            if not isinstance(items, list):
                warnings.append("manifest.yaml must be a list of entries; ignoring it")
                items = []
        except yaml.YAMLError as exc:
            warnings.append(f"manifest.yaml could not be read ({exc}); falling back to file names")
    listed = set()
    for i, it in enumerate(items, start=1):
        if isinstance(it, dict) and all(v in (None, "", []) for v in it.values()):
            continue  # an empty template entry
        if not isinstance(it, dict) or not it.get("image"):
            warnings.append(f"manifest entry {i}: needs at least an 'image' field; skipped")
            continue
        img = raw_dir / str(it["image"])
        listed.add(img.name)
        if not img.exists():
            warnings.append(f"manifest entry {i}: {img.name} not found in ground_truth/raw; skipped")
            continue
        ann = ann_dir / str(it["annotation"]) if it.get("annotation") else None
        if ann is not None and not ann.exists():
            warnings.append(f"manifest entry {i}: annotation {ann.name} not found; using counts only")
            ann = None
        counts = it.get("pod_counts")
        if counts is not None and (not isinstance(counts, list) or not all(isinstance(c, int) and c >= 0 for c in counts)):
            warnings.append(f"manifest entry {i}: pod_counts must be a list of whole numbers; ignored")
            counts = None
        if ann is None and counts is None:
            warnings.append(f"manifest entry {i}: neither a usable annotation nor pod_counts; skipped")
            continue
        if counts is None:
            warnings.append(f"{img.name}: no pod_counts; seed counts from the outlines are used as truth")
        entries.append(GTEntry(img, ann, counts))
    # raw images without a manifest entry: pair with an annotation of the same file stem
    if raw_dir.is_dir():
        for img in sorted(raw_dir.iterdir()):
            if img.suffix.lower() not in (".tif", ".tiff") or img.name in listed:
                continue
            match = [a for a in ann_dir.glob(img.stem + ".*") if a.suffix.lower() in ANNOTATION_EXTS] if ann_dir.is_dir() else []
            if match:
                warnings.append(f"{img.name}: not in manifest.yaml; paired with {match[0].name} by file name")
                entries.append(GTEntry(img, match[0], None))
            else:
                warnings.append(f"{img.name}: not in manifest.yaml and no annotation with the same name; skipped")
    return entries, warnings


def extract_outlines(raw_rgb: np.ndarray, ann_rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Label image (full raw resolution) with one label per annotated seed."""
    warnings = []
    H, W = raw_rgb.shape[:2]
    h, w = ann_rgb.shape[:2]
    if abs((w / h) / (W / H) - 1) > 0.01:
        raise ValueError(f"annotation aspect ratio {w}x{h} does not match the raw image {W}x{H}; "
                         "annotate the full, uncropped frame")
    raw_small = cv2.resize(raw_rgb, (w, h), interpolation=cv2.INTER_AREA).astype(int)
    ann = ann_rgb.astype(int)
    hsv = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2HSV)
    pods = cv2.resize(pod_mask(raw_rgb, cfg).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    whiter = (ann[..., 2] - raw_small[..., 2] > 40) & (ann[..., 2] > 150)        # blue rises under white ink
    white = (hsv[..., 1] < 50) & (hsv[..., 2] > 200)
    stroke = (whiter | white) & ndi.binary_dilation(pods, iterations=2)
    stroke = ndi.binary_closing(stroke, iterations=1)
    inside = ndi.binary_fill_holes(stroke) & ~stroke
    lab = label(inside)
    props = regionprops(lab)
    big = [p.area for p in props if p.area > 50]
    if not big:
        warnings.append("no closed white outlines found")
        return np.zeros((H, W), np.int32), warnings
    med = float(np.median(big))
    keep = np.zeros(lab.max() + 1, bool)
    for p in props:
        keep[p.label] = p.area >= 0.3 * med and p.solidity > 0.85
    lab[~keep[lab]] = 0
    lab = label(lab > 0)
    full = cv2.resize(lab.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    # the drawn line sits on the seed border; grow the interiors by half a line width
    grow = max(1, int(round(W / w)))
    full = np.where(full > 0, full, ndi.grey_dilation(full, size=(2 * grow + 1, 2 * grow + 1)))
    return full, warnings


def load_ground_truth(entry: GTEntry, cfg: dict):
    """(raw_rgb, gt_labels or None, warnings)"""
    raw = load_image(entry.image)
    if entry.annotation is None:
        return raw, None, []
    labels, warns = extract_outlines(raw, load_image(entry.annotation), cfg)
    return raw, labels, warns
