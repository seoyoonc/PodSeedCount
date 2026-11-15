"""Keyboard shortcuts of the review page, read from config/keybindings.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

# action -> (default keys, description shown in the page)
ACTIONS = {
    "accept": (["A"], "accept / add as seed"),
    "reject": (["R"], "reject / not a seed"),
    "restore": (["U"], "undo decision"),
    "delete": (["Delete", "Backspace"], "delete added seed"),
    "next": (["N"], "next to check"),
    "previous": (["Shift+N"], "previous to check"),
    "previous_image": (["["], "previous image"),
    "next_image": (["]"], "next image"),
    "overview": (["O"], "overview"),
    "select_mode": (["S"], "select mode"),
    "add_mode": (["E"], "add seed mode"),
    "split_mode": (["X"], "split seed mode"),
    "redraw_mode": (["D"], "redraw border"),
    "cancel": (["Escape"], "cancel / clear selection"),
    "fit": (["F"], "fit image"),
    "toggle_checked": (["C"], "image checked"),
    "undo": (["Ctrl+Z"], "undo last change"),
}
MODIFIERS = ("Ctrl", "Alt", "Shift")
ALIASES = {"del": "Delete", "delete": "Delete", "backspace": "Backspace", "esc": "Escape", "escape": "Escape",
           "enter": "Enter", "return": "Enter", "space": "Space", "spacebar": "Space", "tab": "Tab",
           "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
           "left": "ArrowLeft", "right": "ArrowRight", "up": "ArrowUp", "down": "ArrowDown",
           "arrowleft": "ArrowLeft", "arrowright": "ArrowRight", "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
           "control": "Ctrl", "ctrl": "Ctrl", "cmd": "Ctrl", "command": "Ctrl", "meta": "Ctrl",
           "alt": "Alt", "option": "Alt", "shift": "Shift"}
NAMED = {"Delete", "Backspace", "Escape", "Enter", "Space", "Tab", "Home", "End", "PageUp", "PageDown",
         "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"} | {f"F{i}" for i in range(1, 13)}


def normalize(combo) -> str:
    """'ctrl+z' -> 'Ctrl+Z', 'del' -> 'Delete', 'n' -> 'N'. Raises ValueError for unusable keys."""
    text = str(combo).strip()
    if not text:
        raise ValueError("empty key")
    if text == "+":
        parts = ["+"]
    else:
        parts = [p.strip() for p in (text[:-1].split("+") + ["+"] if text.endswith("++") else text.split("+"))]
    if any(not p for p in parts):
        raise ValueError(f"'{combo}' is not a valid key")
    *mods, key = parts
    mods = {ALIASES.get(m.lower(), m) for m in mods}
    bad = [m for m in mods if m not in MODIFIERS]
    if bad:
        raise ValueError(f"'{combo}': unknown modifier {bad[0]} (use Ctrl, Alt or Shift)")
    if len(key) == 1:
        if key == " ":
            key = "Space"
        elif not key.isprintable():
            raise ValueError(f"'{combo}' is not a valid key")
        key = key.upper()
        if "Shift" in mods and not key.isalpha():
            raise ValueError(f"'{combo}': write the character itself (e.g. \"?\") instead of Shift+ a character")
    else:
        key = ALIASES.get(key.lower(), key[:1].upper() + key[1:])
        if key in MODIFIERS:
            raise ValueError(f"'{combo}': a modifier on its own cannot be a shortcut")
        if key not in NAMED:
            raise ValueError(f"'{combo}': unknown key name {key}")
    return "+".join([m for m in MODIFIERS if m in mods] + [key])


def load(path: str | Path | None) -> tuple[dict, list[str]]:
    """{action: [keys]} with defaults for missing actions, plus warnings. Never fails."""
    warnings: list[str] = []
    user: dict = {}
    if path and Path(path).exists():
        try:
            user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            if not isinstance(user, dict):
                warnings.append("keybindings.yaml must map actions to keys; using the default keys")
                user = {}
        except yaml.YAMLError as exc:
            warnings.append(f"keybindings.yaml could not be read ({exc}); using the default keys")
            user = {}
    used: dict[str, str] = {}
    out: dict[str, list[str]] = {}
    # 1. keys set in the file (they win over defaults)
    for action, keys in user.items():
        if action not in ACTIONS:
            warnings.append(f"keybindings.yaml: unknown action '{action}' ignored "
                            f"(known: {', '.join(ACTIONS)})")
            continue
        if keys is None:
            keys = []
        if not isinstance(keys, list):
            keys = [keys]
        out[action] = []
        for k in keys:
            try:
                n = normalize(k)
            except ValueError as exc:
                warnings.append(f"keybindings.yaml, {action}: {exc}")
                continue
            if n in used:
                warnings.append(f"keybindings.yaml: {n} is used for both '{used[n]}' and '{action}'; "
                                f"kept for '{used[n]}'")
                continue
            used[n] = action
            out[action].append(n)
    # 2. defaults for actions the file does not mention
    for action, (defaults, _) in ACTIONS.items():
        if action in out:
            continue
        out[action] = []
        for n in defaults:
            if n in used:
                warnings.append(f"default key {n} of '{action}' is now used for '{used[n]}'; "
                                f"'{action}' has no key unless you give it one")
                continue
            used[n] = action
            out[action].append(n)
    return out, warnings


def descriptions() -> dict:
    return {a: d for a, (_, d) in ACTIONS.items()}
