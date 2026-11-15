"""Reading images safely on Windows and Linux (non-ASCII paths, multi-page and 16-bit TIFFs)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


def list_images(folder: str | Path, extensions=(".tif", ".tiff"), recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    pattern = "**/*" if recursive else "*"
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in exts)


def to_rgb8(arr: np.ndarray) -> np.ndarray:
    """Any TIFF page -> H x W x 3 uint8 RGB."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)  # planar -> interleaved
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        hi = np.percentile(a, 99.9) if a.max() > 255 else 255.0
        a = np.clip(a / max(hi, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def load_tif(path: str | Path) -> np.ndarray:
    """Load the largest page of a (possibly multi-page) TIFF as RGB uint8."""
    with tifffile.TiffFile(str(path)) as tif:
        pages = list(tif.pages)
        best = max(pages, key=lambda p: int(np.prod(p.shape[:2])) if len(p.shape) >= 2 else 0)
        return to_rgb8(best.asarray())


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return load_tif(path)
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII paths on Windows
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_png(path: str | Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise IOError(f"Could not encode {path}")
    Path(path).write_bytes(buf.tobytes())


def jpeg_bytes(rgb: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
