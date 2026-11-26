"""Which settings differ from the shipped ones, with the description from the settings file.

``config/default.yaml`` is the live settings file: "Learn from reviews" and calibrate.py --apply write
into it. ``shipped_settings.yaml`` (next to this module) is an untouched copy of the settings as they
were released, so the review page can show what has been changed and restore it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml

SHIPPED = Path(__file__).resolve().parent / "shipped_settings.yaml"
COUNTING_KEYS = ("processing", "pods", "contrast", "segmentation")   # what decides seeds and counts


def fingerprint(cfg: dict) -> str:
    """Short code for everything that decides the counts: the counting settings, the confidence model and
    the analysis version. Same program version + same fingerprint = identical counts on any computer.
    (Thresholds and output options are left out: they change colours, flags and files, not counts.)"""
    from . import ANALYSIS_VERSION
    part = {k: cfg[k] for k in COUNTING_KEYS}
    part["seed_model"] = cfg["confidence"]["seed_model"]
    part["version"] = ANALYSIS_VERSION
    return hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]


def shipped_config() -> dict:
    return yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
KEY_LINE = re.compile(r"^(\s*)([A-Za-z_][\w]*):(.*)$")


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def descriptions(text: str) -> dict:
    """dotted key -> the comment written after it in a YAML settings file (continuation lines joined)."""
    out, stack, last = {}, [], None
    for line in text.splitlines():
        m = KEY_LINE.match(line)
        if m:
            indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, key))
            dotted = ".".join(k for _, k in stack)
            comment = rest.split("#", 1)[1].strip() if "#" in rest and not rest.strip().startswith(("'", '"')) else ""
            out[dotted] = comment
            last = (dotted, indent)
            continue
        s = line.strip()
        if s.startswith("#") and last and len(line) - len(line.lstrip()) > last[1] + 2:
            out[last[0]] = (out[last[0]] + " " + s[1:].strip()).strip()   # comment continued on the next line
        elif s:
            last = None
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def compare(cfg: dict, config_file: str | Path | None = None) -> dict:
    """Settings in use vs the standard settings of this version, plus the backup file if there is one."""
    from . import __version__
    shipped_text = SHIPPED.read_text(encoding="utf-8")
    std_cfg = yaml.safe_load(shipped_text)
    shipped = flatten(std_cfg)
    desc = descriptions(shipped_text)
    current = flatten(cfg)
    rows = []
    for key in sorted(set(shipped) | set(current), key=lambda k: (k.split(".")[0], k)):
        sv, cv = shipped.get(key), current.get(key)
        rows.append({"key": key, "section": key.split(".")[0], "standard": _fmt(sv) if key in shipped else "-",
                     "current": _fmt(cv) if key in current else "-", "changed": sv != cv,
                     "description": desc.get(key, "")})
    backup = None
    if config_file:
        bpath = Path(config_file).with_suffix(".backup.yaml")
        if bpath.exists():
            try:
                old = flatten(yaml.safe_load(bpath.read_text(encoding="utf-8")) or {})
                diffs = [{"key": k, "backup": _fmt(old[k]), "current": _fmt(current[k]), "description": desc.get(k, "")}
                         for k in sorted(old) if k in current and old[k] != current[k]]
                backup = {"file": str(bpath), "saved": dt.datetime.fromtimestamp(bpath.stat().st_mtime).isoformat(timespec="minutes"),
                          "differences": diffs}
            except Exception:
                backup = None
    fp = fingerprint(cfg)
    std_fp = fingerprint(_complete(std_cfg, cfg))
    return {"version": __version__, "fingerprint": fp, "standard_fingerprint": std_fp,
            "counting_is_standard": fp == std_fp, "rows": rows, "changed": sum(r["changed"] for r in rows),
            "file": str(config_file or ""), "backup": backup}


def _complete(std: dict, cfg: dict) -> dict:
    """Standard settings with any section missing from the shipped copy taken from cfg (for the fingerprint)."""
    out = dict(cfg)
    out.update(std)
    return out


def reset_to_shipped(path: str | Path) -> Path:
    """Overwrite the settings file with the shipped settings; the old file is kept as *.backup.yaml."""
    path = Path(path)
    backup = path.with_suffix(".backup.yaml")
    shutil.copy(path, backup)
    shutil.copy(SHIPPED, path)
    return backup


# ------------------------------------------------------------------ plain-language catalogue
def _pct(v):
    return f"{float(v) * 100:g} %"


def _yes(v):
    return "yes" if v else "no"


# Each topic: title, what it is about, which settings belong to it, and the few settings that matter most,
# as (key, plain name, plain explanation, how to show the value). Everything else is listed as technical.
CATALOG = [
    {"id": "pods", "title": "Pods", "about": "Finding pods against the background.",
     "match": lambda k: k.startswith("pods.") and k.split(".")[1] not in ("left_body_frac", "right_body_frac", "bud_frac", "end_margin", "rim_frac"),
     "main": [
         ("pods.colour_threshold", "Detection threshold", "Lower finds paler pods, but also more dust.",
          lambda v: f"{v:g}"),
         ("pods.split_touching", "Split touching pods", "", _yes),
         ("pods.min_area_frac", "Minimum pod size", "Smaller objects are ignored. Relative to the largest pod.", _pct),
     ]},
    {"id": "ends", "title": "Counted area", "about": "Where counting starts and stops in each pod.",
     "match": lambda k: k.split(".")[-1] in ("left_body_frac", "right_body_frac", "bud_frac", "end_margin", "rim_frac"),
     "main": [
         ("pods.left_body_frac", "Bud end", "Counting starts where the pod reaches this width.",
          lambda v: f"{_pct(v)} of pod width"),
         ("pods.right_body_frac", "Stem end", "Counting stops where the pod narrows below this width.",
          lambda v: f"{_pct(v)} of pod width"),
     ]},
    {"id": "size", "title": "Seed size", "about": "",
     "match": lambda k: k.startswith("processing."),
     "main": [
         ("processing.seed_radius_px", "Seed radius", "At full image size.", lambda v: f"{v:g} px"),
         ("processing.auto_radius", "Adjust per image", "Re-measure when seeds are clearly bigger or smaller.", _yes),
     ]},
    {"id": "seeds", "title": "Seeds", "about": "Finding and separating seeds.",
     "match": lambda k: k.startswith(("contrast.", "segmentation.")),
     "main": [
         ("contrast.channel", "Colour channel", "", lambda v: {"green": "green", "saturation": "saturation"}.get(v, str(v))),
         ("segmentation.ellipse.surround_weight", "Packed seeds", "Higher separates tightly packed seeds better.",
          lambda v: f"{v:g}"),
         ("segmentation.ellipse.relative_level", "Pale spot cut-off", "Spots paler than this (vs. a typical seed) are not counted.", _pct),
         ("segmentation.ellipse.max_overlap", "Max overlap", "More overlap than this counts as one seed.", _pct),
         ("segmentation.ellipse.max_angle_to_pod", "Max seed angle", "Relative to the pod.", lambda v: f"{v:g}°"),
     ]},
    {"id": "confidence", "title": "Colours and flags", "about": "Seed colours and review flags. Does not change counts.",
     "match": lambda k: k.startswith("confidence."),
     "main": [
         ("confidence.review_threshold", "Green from", "", _pct),
         ("confidence.reject_threshold", "Red below", "", _pct),
     ]},
    {"id": "learning", "title": "Learning", "about": "Used by Learn from reviews.",
     "match": lambda k: k.startswith("calibration."),
     "main": []},
    {"id": "output", "title": "Output", "about": "Files written by Save results. Does not change counts.",
     "match": lambda k: k.startswith(("output.", "render.")),
     "main": [
         ("output.annotated_png", "Annotated images", "", _yes),
         ("output.seeds_csv", "Seed table", "One row per seed.", _yes),
         ("render.scale", "Image size", "Relative to the original.", _pct),
     ]},
    {"id": "input", "title": "Input", "about": "",
     "match": lambda k: k.startswith("input."),
     "main": [
         ("input.extensions", "File types", "", lambda v: ", ".join(v)),
         ("input.recursive", "Include sub-folders", "", _yes),
     ]},
]


def plain_groups(cfg: dict) -> list[dict]:
    """The settings grouped into topics, with plain names for the main ones and the rest as technical."""
    shipped_text = SHIPPED.read_text(encoding="utf-8")
    std = flatten(yaml.safe_load(shipped_text))
    desc = descriptions(shipped_text)
    cur = flatten(cfg)
    keys = sorted(set(std) | set(cur))
    out, seen = [], set()
    for g in CATALOG:
        main_keys = [m[0] for m in g["main"]]
        rows_main, rows_tech = [], []
        for key, name, explain, show in g["main"]:
            if key not in cur and key not in std:
                continue
            seen.add(key)
            sv, cv = std.get(key), cur.get(key)

            def txt(v):
                try:
                    return show(v) if v is not None else "-"
                except Exception:
                    return _fmt(v)
            rows_main.append({"key": key, "name": name, "explain": explain, "standard": txt(sv), "current": txt(cv),
                              "changed": sv != cv})
        for key in keys:
            if key in seen or key in main_keys or not g["match"](key):
                continue
            seen.add(key)
            sv, cv = std.get(key), cur.get(key)
            rows_tech.append({"key": key, "name": key, "explain": desc.get(key, ""), "standard": _fmt(sv) if key in std else "-",
                              "current": _fmt(cv) if key in cur else "-", "changed": sv != cv})
        out.append({"id": g["id"], "title": g["title"], "about": g["about"], "main": rows_main, "technical": rows_tech,
                    "changed": sum(r["changed"] for r in rows_main + rows_tech)})
    rest = [k for k in keys if k not in seen]
    if rest:
        out.append({"id": "other", "title": "Other", "about": "", "main": [], "changed": sum(std.get(k) != cur.get(k) for k in rest),
                    "technical": [{"key": k, "name": k, "explain": desc.get(k, ""), "standard": _fmt(std.get(k)),
                                   "current": _fmt(cur.get(k)), "changed": std.get(k) != cur.get(k)} for k in rest]})
    return out
