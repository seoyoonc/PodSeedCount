"""Review-page features on a synthetic image (no example images needed): manual review flags, redrawing and
drawing pods, removing and restoring images, saving selected images, and the settings comparison."""
import json

import cv2
import numpy as np
import pytest
import tifffile

from seedcounter.config import load_config
from seedcounter.review.session import ReviewSession
from seedcounter.settings_info import SHIPPED, compare, descriptions, flatten
from seedcounter.summary import image_summary

BG = (252, 250, 254)


def _image(path, pods=((80, 400), (210, 400))):
    """White image with long yellow pods (bud left, stem right) holding two rows of brown seeds."""
    img = np.full((300, 1000, 3), BG, np.uint8)
    for y, half in pods:
        cv2.ellipse(img, (500, y), (half, 38), 0, 0, 360, (236, 226, 170), -1)
        cv2.rectangle(img, (500 + half - 20, y - 6), (990, y + 6), (230, 222, 175), -1)
        for x in range(260, 760, 46):
            for dy in (-15, 15):
                cv2.ellipse(img, (x + (23 if dy > 0 else 0), y + dy), (17, 12), 0, 0, 360, (165, 120, 45), -1)
    tifffile.imwrite(str(path), cv2.GaussianBlur(img, (0, 0), 1.0))


@pytest.fixture
def session(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    _image(inp / "a.tif")
    _image(inp / "b.tif", pods=((150, 400),))
    cfg = load_config(overrides={"processing.seed_radius_px": 14, "processing.auto_radius": False,
                                 "processing.work_scale": 1.0})
    s = ReviewSession(inp, tmp_path / "out", cfg, "")
    s.load_all()
    return s


@pytest.mark.nodata
def test_manual_flags_override_and_replay(session, tmp_path):
    cfg = session.cfg
    session.apply("a.tif", {"op": "flag_pod", "pod": 1, "value": "maybe"})
    auto = image_summary(session.current["a.tif"], cfg)["needs_review"]   # follows the pods, incl. pod flags
    assert auto in ("maybe", "yes")
    manual = "no"
    session.apply("a.tif", {"op": "flag", "value": manual})
    s = image_summary(session.current["a.tif"], cfg)
    assert s["needs_review"] == manual and s["auto_needs_review"] == auto and s["flag_source"] == "reviewer"
    p1 = next(r for r in s["pod_rows"] if r["pod"] == 1)
    assert p1["needs_review"] == "maybe" and p1["flag_source"] == "reviewer"
    session.apply("a.tif", {"op": "reviewed", "value": True})      # a manual flag wins over "checked"
    assert image_summary(session.current["a.tif"], cfg)["needs_review"] == manual
    session.apply("a.tif", {"op": "flag", "value": None})
    assert image_summary(session.current["a.tif"], cfg)["flag_source"] == "automatic"
    again = ReviewSession(session.input_dir, session.output_dir, cfg, "")
    again.load_all()
    assert image_summary(again.current["a.tif"], cfg)["pod_rows"][0]["needs_review"] == "maybe"
    with pytest.raises(ValueError):
        session.apply("a.tif", {"op": "flag", "value": "perhaps"})


@pytest.mark.nodata
def test_redraw_pod_recounts_and_absorbs_pieces(session):
    k = "a.tif"
    img = session.current[k]
    pod = img.active_pods()[0]
    n_before = len(pod.active_seeds())
    full = pod.outline
    # make the pod look broken: shrink pod 1 to its right half, as if the pale bud half were lost
    xs = np.asarray(full)[:, 0]
    mid = (xs.min() + xs.max()) / 2
    half = [p for p in full if p[0] >= mid]
    session.apply(k, {"op": "redraw_pod", "pod": pod.index, "outline": half})
    assert len(session.current[k].pod(pod.index).active_seeds()) < n_before
    # draw the lost half as a new pod, then redraw pod 1 with its full outline: the piece stops counting
    left = [p for p in full if p[0] <= mid + 5]
    session.apply(k, {"op": "add_pod", "outline": left})
    added = max(p.index for p in session.current[k].pods)
    assert session.current[k].pod(added).active_seeds()
    session.apply(k, {"op": "redraw_pod", "pod": pod.index, "outline": full})
    now = session.current[k]
    assert now.pod(added).status == "ignored"
    assert abs(len(now.pod(pod.index).active_seeds()) - n_before) <= 1
    ids = [s.id for p in now.pods for s in p.seeds]
    assert len(ids) == len(set(ids))
    # the stored analysis makes replay exact, without the image
    again = ReviewSession(session.input_dir, session.output_dir, session.cfg, "")
    again.load_all()
    assert [len(p.active_seeds()) for p in again.current[k].active_pods()] == [len(p.active_seeds()) for p in now.active_pods()]
    # undo restores the piece
    session.undo(k)
    assert session.current[k].pod(added).status == "active"


@pytest.mark.nodata
def test_redraw_without_recount_keeps_seeds(session):
    k = "a.tif"
    pod = session.current[k].active_pods()[0]
    ids = sorted(s.id for s in pod.seeds)
    session.apply(k, {"op": "redraw_pod", "pod": pod.index, "outline": pod.outline, "recount": False})
    assert sorted(s.id for s in session.current[k].pod(pod.index).seeds) == ids


@pytest.mark.nodata
def test_remove_restore_and_selective_save(session):
    session.apply("a.tif", {"op": "flag", "value": "yes"})
    r = session.export(session.cfg, ["b.tif"])
    assert r["images"] == 1 and r["of"] == 2
    rows = (session.output_dir / "output_table" / "run_summary.csv").read_text(encoding="utf-8-sig")
    assert "b.tif" in rows and "a.tif" not in rows
    with pytest.raises(ValueError):
        session.export(session.cfg, [])
    session.remove("a.tif")
    assert session.keys == ["b.tif"]
    assert not (session.input_dir / "a.tif").exists()
    assert not list((session.output_dir / "review" / "sessions").glob("a.tif*"))   # no longer used for learning
    assert [r["key"] for r in session.removed()] == ["a.tif"]
    session.restore("a.tif")
    assert set(session.keys) == {"a.tif", "b.tif"} and (session.input_dir / "a.tif").exists()
    assert image_summary(session.current["a.tif"], session.cfg)["needs_review"] == "yes"   # edits came back
    assert session.removed() == []


@pytest.mark.nodata
def test_settings_comparison(tmp_path):
    cfg = load_config(overrides={"pods.right_body_frac": 0.7})
    rep = compare(cfg, None)
    changed = [r for r in rep["rows"] if r["changed"]]
    assert [r["key"] for r in changed] == ["pods.right_body_frac"]
    assert changed[0]["standard"] == "0.65" and changed[0]["current"] == "0.7" and changed[0]["description"]


@pytest.mark.nodata
def test_fingerprint_tracks_counting_settings_only():
    from seedcounter.settings_info import fingerprint
    base = fingerprint(load_config())
    assert fingerprint(load_config(overrides={"confidence.review_threshold": 0.8})) == base   # colours only
    assert fingerprint(load_config(overrides={"output.annotated_png": False})) == base
    assert fingerprint(load_config(overrides={"segmentation.peak_sigma": 0.2})) != base
    rep = compare(load_config(), None)
    assert rep["counting_is_standard"] and rep["fingerprint"] == base


@pytest.mark.nodata
def test_shipped_settings_have_the_same_keys_as_the_settings_file():
    import yaml
    from seedcounter.config import DEFAULT_CONFIG
    shipped = flatten(yaml.safe_load(SHIPPED.read_text(encoding="utf-8")))
    live = flatten(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert set(shipped) == set(live)
    d = descriptions(SHIPPED.read_text(encoding="utf-8"))
    assert d["pods.colour_threshold"].startswith("colour")


@pytest.mark.nodata
def test_every_setting_is_in_a_topic():
    from seedcounter.settings_info import plain_groups
    groups = plain_groups(load_config())
    keys = [r["key"] for g in groups for r in g["main"] + g["technical"]]
    assert sorted(keys) == sorted(flatten(load_config()))
    assert all(g["id"] != "other" for g in groups)


@pytest.mark.nodata
def test_load_settings_file(session, tmp_path):
    import shutil, time
    from seedcounter.config import DEFAULT_CONFIG
    from seedcounter.calibrate import update_yaml_text
    cfg_file = tmp_path / "settings.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_file)
    session.config_file = str(cfg_file)
    other = update_yaml_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), "pods.right_body_frac", 0.7)
    with pytest.raises(ValueError):
        session.start_load_settings(b"hello: world\n")
    with pytest.raises(ValueError):
        session.start_load_settings(b"pods: {}\nnot_a_setting: 1\n")
    session.start_load_settings(other.encode())
    for _ in range(200):
        if session.job["state"] != "running":
            break
        time.sleep(0.05)
    assert session.job["state"] == "done", session.job
    assert session.cfg["pods"]["right_body_frac"] == 0.7
    assert (tmp_path / "settings.backup.yaml").exists()
