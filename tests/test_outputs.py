import json
import shutil

from conftest import DATA, ROOT
from seedcounter.config import load_config
from seedcounter.run import run_folder


def test_full_run_writes_all_outputs_and_survives_bad_files(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp / "pods ä.tif")   # space + non-ASCII name
    (inp / "broken.tif").write_bytes(b"not a tiff")
    cfg = load_config(overrides={"output.seeds_csv": True, "output.results_json": "full"})
    out = tmp_path / "out"
    run, sums = run_folder(inp, out, cfg)
    by_name = {s["image"]: s for s in sums}
    assert by_name["broken.tif"]["needs_review"] == "error"
    assert by_name["pods ä.tif"]["seeds"] > 100
    for f in ["report.html", "results.json", "output_table/run_summary.csv", "output_table/pods.csv",
              "output_table/report_outline.csv", "output_table/seeds.csv", "annotated_images/pods ä_annotated.png"]:
        assert (out / f).exists(), f
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["run"]["images_processed"] == 2 and "results" in data
    assert "1 could not be processed" in (out / "output_table" / "run_summary.csv").read_text(encoding="utf-8-sig")


def test_run_mode_json_has_no_seed_data(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", inp)
    run_folder(inp, tmp_path / "out", load_config())
    data = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert set(data) == {"run", "images"} and "pod_rows" not in data["images"][0]
