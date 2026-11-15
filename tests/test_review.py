"""Review session and local server: edits persist, undo works, export writes reviewed outputs."""
import json
import shutil
import threading
import urllib.error
import urllib.request

import pytest

from conftest import DATA, ROOT
from seedcounter.config import load_config
from seedcounter.review.server import serve
from seedcounter.review.session import ReviewSession


@pytest.fixture()
def session(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp)
    s = ReviewSession(inp, tmp_path / "out", load_config())
    s.load_all()
    return s


def test_edits_persist_undo_and_reload(session):
    key = session.keys[0]
    cfg = session.config_with()
    n0 = session.payload(0, cfg)["summary"]["seeds"]
    seed = session.current[key].pods[0].seeds[0]
    session.apply(key, {"op": "reject", "seed": seed.id})
    session.apply(key, {"op": "reviewed", "value": True})
    p = session.payload(0, cfg)
    assert p["summary"]["seeds"] == n0 - 1 and p["reviewed"] and p["summary"]["needs_review"] == "no"
    with pytest.raises(ValueError):
        session.apply(key, {"op": "add", "pod": 1, "contour": [[0, 0]]})
    assert len(session.edits[key]) == 2                     # invalid edit not recorded
    # a new session (e.g. after a restart) sees the same state
    again = ReviewSession(session.input_dir, session.output_dir, load_config())
    again.load_all()
    assert again.payload(0, cfg)["summary"]["seeds"] == n0 - 1
    again.undo(key)
    again.undo(key)
    assert again.payload(0, cfg)["summary"]["seeds"] == n0 and not again.edits[key]


def test_threshold_change_needs_no_reprocessing(session):
    loose = session.payload(0, session.config_with(0.5, 0.2))["summary"]
    strict = session.payload(0, session.config_with(0.99, 0.9))["summary"]
    assert loose["seeds"] == strict["seeds"] and strict["green"] < loose["green"]


def test_export_writes_reviewed_results(session):
    key = session.keys[0]
    session.apply(key, {"op": "reviewed", "value": True})
    r = session.export(session.config_with())
    out = session.output_dir
    assert r["reviewed"] == 1 and (out / "report.html").exists()
    text = (out / "output_table" / "pods.csv").read_text(encoding="utf-8-sig")
    assert "checked by reviewer" in text


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=20).read())


def test_server_endpoints(session):
    httpd = serve(session, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert b"Seed review" in urllib.request.urlopen(base + "/", timeout=20).read()
        lst = json.loads(urllib.request.urlopen(base + "/api/images?review=0.8&reject=0.3", timeout=20).read())
        assert lst["thresholds"] == {"review": 0.8, "reject": 0.3} and len(lst["images"]) == 1
        assert urllib.request.urlopen(base + "/api/image/0/picture", timeout=20).read()[:2] == b"\xff\xd8"
        seed_id = json.loads(urllib.request.urlopen(base + "/api/image/0", timeout=20).read())["pods"][0]["seeds"][0]["id"]
        assert _post(base + "/api/image/0/edit", {"op": "reject", "seed": seed_id})["edits"] == 1
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/image/0/edit", {"op": "explode"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/api/image/7", timeout=20)
        assert e.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
