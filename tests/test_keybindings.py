import pytest
from conftest import ROOT
from seedcounter.review import keybindings as kb


@pytest.mark.nodata
def test_shipped_file_matches_defaults_without_warnings():
    binds, warns = kb.load(ROOT / "config" / "keybindings.yaml")
    assert warns == []
    assert binds == {a: d for a, (d, _) in kb.ACTIONS.items()}


@pytest.mark.nodata
def test_normalize():
    assert kb.normalize("ctrl+z") == "Ctrl+Z" and kb.normalize("shift+ctrl+z") == "Ctrl+Shift+Z"
    assert kb.normalize("del") == "Delete" and kb.normalize("esc") == "Escape" and kb.normalize("n") == "N"
    assert kb.normalize("[") == "[" and kb.normalize("cmd+s") == "Ctrl+S" and kb.normalize("f2") == "F2"
    for bad in ("Shift+/", "Hyper+A", "Shift", "Banana", ""):
        try:
            kb.normalize(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


@pytest.mark.nodata
def test_user_keys_win_and_conflicts_are_reported(tmp_path):
    f = tmp_path / "k.yaml"
    f.write_text("accept: [K]\nreject: [R, k]\nsplit_mode: [D]\nfly: [Q]\nfit: []\nnext: [Shift+/]\n", encoding="utf-8")
    binds, warns = kb.load(f)
    assert binds["accept"] == ["K"] and binds["reject"] == ["R"]          # K kept for the first action
    assert binds["split_mode"] == ["D"] and binds["redraw_mode"] == []    # default D given away
    assert binds["fit"] == [] and binds["next"] == [] and binds["undo"] == ["Ctrl+Z"]
    text = " ".join(warns)
    assert "fly" in text and "both" in text and "redraw_mode" in text and "Shift+/" in text


@pytest.mark.nodata
def test_broken_file_falls_back_to_defaults(tmp_path):
    f = tmp_path / "k.yaml"
    f.write_text("accept: [A\n", encoding="utf-8")
    binds, warns = kb.load(f)
    assert binds["accept"] == ["A"] and warns
    assert kb.load(tmp_path / "missing.yaml") == ({a: d for a, (d, _) in kb.ACTIONS.items()}, [])
