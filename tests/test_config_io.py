import numpy as np
import pytest

from conftest import DATA, ROOT
from seedcounter.calibrate import update_yaml_text
from seedcounter.config import load_config
from seedcounter.io import list_images, load_tif, to_rgb8


@pytest.mark.nodata
def test_overrides_and_validation():
    cfg = load_config(overrides={"confidence.review_threshold": "0.8", "output.seeds_csv": "true"})
    assert cfg["confidence"]["review_threshold"] == 0.8 and cfg["output"]["seeds_csv"] is True
    with pytest.raises(ValueError):
        load_config(overrides={"confidence.reject_threshold": 0.9, "confidence.review_threshold": 0.5})


@pytest.mark.nodata
def test_yaml_update_keeps_comments():
    text = "a:\n  b: 1          # comment\n  c:\n    b: 2  # inner\n"
    out = update_yaml_text(text, "a.c.b", 5)
    assert "    b: 5  # inner" in out and "  b: 1          # comment" in out


def test_multipage_tif_uses_largest_page():
    img = load_tif(DATA / "ground_truth" / "raw" / "d98bd67c-1_Col_0002.tif")
    assert img.shape == (2048, 2880, 3) and img.dtype == np.uint8


@pytest.mark.nodata
def test_to_rgb8_handles_16bit_and_gray():
    a = (np.arange(100, dtype=np.uint16).reshape(10, 10) * 600)
    out = to_rgb8(a)
    assert out.shape == (10, 10, 3) and out.dtype == np.uint8 and out.max() == 255


@pytest.mark.nodata
def test_list_images(tmp_path):
    (tmp_path / "a.TIF").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["a.TIF"]
