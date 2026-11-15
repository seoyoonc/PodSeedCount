import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"      # the example images and their annotations, used only by the tests
HAVE_DATA = (DATA / "ground_truth" / "manifest.yaml").exists() and any((DATA / "ground_truth" / "raw").glob("*.tif"))
sys.path.insert(0, str(ROOT / "src"))

from seedcounter.config import load_config  # noqa: E402
from seedcounter.ground_truth import load_ground_truth, load_manifest  # noqa: E402
from seedcounter.pipeline import process_image  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "nodata: test does not need the example images in tests/data")


@pytest.fixture(autouse=True)
def _needs_example_images(request):
    """Tests that use the example images are skipped (not failed) when tests/data is absent."""
    if not HAVE_DATA and request.node.get_closest_marker("nodata") is None:
        pytest.skip("example images not present in tests/data")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def gt_results(cfg):
    """(entry, gt_labels, result) for every example ground-truth image, computed once."""
    entries, _ = load_manifest(DATA / "ground_truth")
    out = []
    for e in entries:
        _, labels, _ = load_ground_truth(e, cfg)
        out.append((e, labels, process_image(e.image, cfg)))
    return out
