import pytest
import shutil

from conftest import DATA, ROOT
from seedcounter.ground_truth import load_manifest


def test_manifest_loads_examples():
    entries, warnings = load_manifest(DATA / "ground_truth")
    assert len(entries) >= 2 and all(e.annotation is not None for e in entries)


def test_broken_entries_do_not_stop_others(tmp_path):
    gt = tmp_path / "gt"
    (gt / "raw").mkdir(parents=True)
    (gt / "annotations").mkdir()
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", gt / "raw")
    shutil.copy(DATA / "ground_truth" / "raw" / "09_1829px6_06.tif", gt / "raw" / "extra.tif")
    shutil.copy(DATA / "ground_truth" / "annotations" / "IMG_4318.jpeg", gt / "annotations")
    shutil.copy(DATA / "ground_truth" / "annotations" / "IMG_4318.jpeg", gt / "annotations" / "extra.jpg")
    (gt / "manifest.yaml").write_text(
        "- image: 09_1829px6_06.tif\n  annotation: IMG_4318.jpeg\n  pod_counts: [27, 19, 20, 32, 23, 24]\n"
        "- image: missing.tif\n  pod_counts: [1]\n"
        "- {annotation: x.jpg}\n"
        "- image: 09_1829px6_06.tif\n  annotation: nope.png\n  pod_counts: [1, two]\n", encoding="utf-8")
    entries, warnings = load_manifest(gt)
    names = sorted(e.image.name for e in entries)
    assert names == ["09_1829px6_06.tif", "extra.tif"]           # valid entry + paired by file name
    assert any("missing.tif" in w for w in warnings)
    assert any("pod_counts" in w for w in warnings)


def test_outline_counts_match_hand_counts(gt_results, cfg):
    from seedcounter.evaluate import evaluate_image
    for e, labels, img in gt_results:
        ev = evaluate_image(img, labels, e.pod_counts, cfg)
        for r in ev["pods"]:
            assert abs(r["outline_count"] - r["truth"]) <= 3, (e.image.name, r)


@pytest.mark.nodata
def test_shipped_template_manifest_is_empty_and_quiet(tmp_path):
    from seedcounter.review.uploads import add_manifest_entry
    assert load_manifest(ROOT / "ground_truth") == ([], [])
    gt = tmp_path / "gt"
    shutil.copytree(ROOT / "ground_truth", gt)
    (gt / "raw" / "a.tif").write_bytes(b"x")
    (gt / "annotations" / "a_annotation.png").write_bytes(b"x")
    add_manifest_entry(gt / "manifest.yaml", "a.tif", "a_annotation.png", [3, 4])
    entries, warnings = load_manifest(gt)
    assert [e.image.name for e in entries] == ["a.tif"] and entries[0].pod_counts == [3, 4] and warnings == []
