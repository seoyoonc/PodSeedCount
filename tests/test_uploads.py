"""Adding images and annotation pairs through the review server's upload functions."""
import shutil

import pytest
import yaml

from conftest import DATA, ROOT
from seedcounter.config import load_config
from seedcounter.ground_truth import load_manifest
from seedcounter.review.session import ReviewSession
from seedcounter.review.uploads import AnnotationDesk, save_input_image, start_analyse_new


@pytest.fixture()
def env(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp)
    gt = tmp_path / "gt"
    (gt / "raw").mkdir(parents=True)
    (gt / "annotations").mkdir()
    (gt / "manifest.yaml").write_text("# my comment\n", encoding="utf-8")
    s = ReviewSession(inp, tmp_path / "out", load_config(), gt_dir=gt)
    s.load_all()
    return s, gt


def _wait(s):
    import time
    for _ in range(200):
        if s.job["state"] != "running":
            return s.job
        time.sleep(0.1)
    raise TimeoutError


def test_add_image_and_analyse(env):
    s, _ = env
    data = (DATA / "ground_truth" / "raw" / "d98bd67c-1_Col_0002.tif").read_bytes()
    assert save_input_image(s, "new pods.tif", data)["status"] == "added"
    assert save_input_image(s, "new pods.tif", data)["status"] == "already there"
    with pytest.raises(ValueError):
        save_input_image(s, "new pods.tif", data[:-10] + b"0" * 10)      # different file, same name
    with pytest.raises(ValueError):
        save_input_image(s, "../evil.tif", b"not an image")               # not readable
    with pytest.raises(ValueError):
        save_input_image(s, "notes.txt", b"x")
    assert not list(s.input_dir.glob("*.part"))
    start_analyse_new(s)
    assert _wait(s)["state"] == "done" and s.keys[-1] == "new pods.tif" and s.keys[0] == "09_1829px6_06.tif"
    assert s.payload(1, s.config_with())["summary"]["seeds"] > 200


def test_annotation_check_save_and_remove(env):
    s, gt = env
    desk = AnnotationDesk(s)
    a = desk.stage(None, "raw", "d98bd67c-1_Col_0002.tif", (DATA / "ground_truth/raw/d98bd67c-1_Col_0002.tif").read_bytes())
    desk.stage(a["token"], "annotation", "IMG_4317.jpeg", (DATA / "ground_truth/annotations/IMG_4317.jpeg").read_bytes())
    r = desk.check(a["token"])
    assert [p["shapes"] for p in r["pods"]] == [56, 54, 64, 48, 46]
    with pytest.raises(ValueError):
        desk.save(a["token"], [58, 54])                                   # one count per pod needed
    saved = desk.save(a["token"], [58, 54, 62, 48, 46])
    assert saved["mismatched_pods"] == [1, 3]
    text = (gt / "manifest.yaml").read_text(encoding="utf-8")
    assert text.startswith("# my comment") and yaml.safe_load(text)[0]["pod_counts"] == [58, 54, 62, 48, 46]
    entries, _ = load_manifest(gt)
    assert len(entries) == 1 and entries[0].annotation.exists()
    # the same image cannot be annotated twice
    b = desk.stage(None, "raw", "d98bd67c-1_Col_0002.tif", (DATA / "ground_truth/raw/d98bd67c-1_Col_0002.tif").read_bytes())
    desk.stage(b["token"], "annotation", "IMG_4317.jpeg", (DATA / "ground_truth/annotations/IMG_4317.jpeg").read_bytes())
    assert desk.check(b["token"])["already_annotated"]
    with pytest.raises(ValueError):
        desk.save(b["token"], [58, 54, 62, 48, 46])
    desk.remove("d98bd67c-1_Col_0002.tif")
    assert load_manifest(gt)[0] == [] and (gt / "removed" / "d98bd67c-1_Col_0002.tif").exists()
    assert (gt / "manifest.yaml").read_text(encoding="utf-8").startswith("# my comment")


def test_mismatched_annotation_shape_is_refused(env):
    s, _ = env
    import cv2
    import numpy as np
    desk = AnnotationDesk(s)
    a = desk.stage(None, "raw", "x.tif", (DATA / "ground_truth/raw/09_1829px6_06.tif").read_bytes())
    ok, buf = cv2.imencode(".png", np.full((100, 300, 3), 255, np.uint8))
    desk.stage(a["token"], "annotation", "x.png", buf.tobytes())
    with pytest.raises(ValueError, match="aspect ratio"):
        desk.check(a["token"])


def test_review_starts_with_no_images_and_accepts_uploads(tmp_path):
    inp = tmp_path / "does_not_exist_yet"
    s = ReviewSession(inp, tmp_path / "out", load_config())
    s.load_all()
    assert s.keys == [] and inp.is_dir() and s.listing(s.config_with()) == []
    assert s.totals([])["images"] == 0
    with pytest.raises(ValueError, match="no images"):
        s.export(s.config_with())
    save_input_image(s, "first.tif", (DATA / "ground_truth" / "raw" / "09_1829px6_06.tif").read_bytes())
    start_analyse_new(s)
    assert _wait(s)["state"] == "done" and s.keys == ["first.tif"]
    assert s.payload(0, s.config_with())["summary"]["seeds"] > 100
