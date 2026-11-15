"""Loading, merging and overriding YAML settings."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Default settings, then the given file (only the keys it contains), then overrides."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    validate(cfg)
    return cfg


def get_dotted(cfg: dict, dotted: str) -> Any:
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    if isinstance(value, str):
        value = yaml.safe_load(value)  # "0.8" -> 0.8, "true" -> True
    node[parts[-1]] = value


def validate(cfg: dict) -> None:
    c = cfg["confidence"]
    if not 0 <= c["reject_threshold"] <= c["review_threshold"] <= 1:
        raise ValueError("confidence thresholds must satisfy 0 <= reject_threshold <= review_threshold <= 1")
    if cfg["output"]["results_json"] not in ("run", "full", "off", False):
        raise ValueError("output.results_json must be 'run', 'full' or 'off'")
    if not 0 < cfg["processing"]["work_scale"] <= 1:
        raise ValueError("processing.work_scale must be in (0, 1]")


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
