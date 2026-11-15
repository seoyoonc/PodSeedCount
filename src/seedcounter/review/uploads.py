"""Adding input images and ground-truth annotations from the review page.

Images are copied into the input folder and analysed in the background.
Annotations are staged first (raw TIFF + annotated copy), checked - the enclosed shapes per pod
are counted and compared with the counts the user types - and only then saved into
ground_truth/ with a manifest entry.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.segmentation import find_boundaries

from ..evaluate import evaluate_image
from ..ground_truth import ANNOTATION_EXTS, extract_outlines, load_manifest
from ..io import jpeg_bytes, list_images, load_image
from ..pipeline import process_image

MAX_UPLOAD = 300 * 1024 * 1024  # 300 MB per file


def safe_name(name: str, exts) -> str:
    """Plain file name (no folders), with an allowed extension."""
    name = Path(str(name).replace("\\", "/")).name.strip()
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    if not name or name.startswith("."):
        raise ValueError("invalid file name")
    if Path(name).suffix.lower() not in exts:
        raise ValueError(f"{name}: only {', '.join(exts)} files are accepted")
    return name


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _same_file(path: Path, data: bytes) -> bool:
    return path.exists() and path.stat().st_size == len(data) and _digest(path.read_bytes()) == _digest(data)


# ------------------------------------------------------------------ input images
def save_input_image(session, name: str, data: bytes) -> dict:
    name = safe_name(name, tuple(session.cfg["input"]["extensions"]))
    if not data:
        raise ValueError(f"{name}: the file is empty")
    target = session.input_dir / name
    if target.exists():
        if _same_file(target, data):
            return {"name": name, "status": "already there"}
        raise ValueError(f"{name}: a different file with this name is already in the input folder; rename it first")
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    try:
        load_image(tmp)  # reject files that are not readable images before they enter the folder
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: not a readable TIFF image ({exc})")
    tmp.replace(target)
    return {"name": name, "status": "added"}


def start_analyse_new(session) -> dict:
    """Analyse images that are in the input folder but not yet in the review (background job)."""
    from ..run import image_key
    paths = list_images(session.input_dir, session.cfg["input"]["extensions"], session.cfg["input"]["recursive"])
    new = [(image_key(p, session.input_dir), p) for p in paths]
    new = [(k, p) for k, p in new if k not in session.paths]

    def fn(progress):
        session.job["message"] = f"analysing {len(new)} new image{'s' if len(new) != 1 else ''}"
        for k, p in progress(new):
            with session.lock:
                session.paths[k] = p
            session._load(k)
            with session.lock:
                session.keys.append(k)  # appended, so the ids of the other images stay the same
        return {"added": len(new)}
    return session._run_job("images", fn)


# ------------------------------------------------------------------ annotations
class AnnotationDesk:
    """Staging, checking and saving of annotation pairs."""

    def __init__(self, session):
        self.session = session
        self.root = session.output_dir / "review" / "staging"
        self.checked: dict[str, dict] = {}

    @property
    def gt_dir(self) -> Path:
        if not self.session.gt_dir:
            raise ValueError("no ground-truth folder: start the review with --gt <folder>")
        return self.session.gt_dir

    def stage(self, token: str | None, role: str, name: str, data: bytes) -> dict:
        if role not in ("raw", "annotation"):
            raise ValueError("role must be 'raw' or 'annotation'")
        exts = (".tif", ".tiff") if role == "raw" else ANNOTATION_EXTS
        name = safe_name(name, exts)
        token = token if token and re.fullmatch(r"[0-9a-f]{32}", token) else uuid.uuid4().hex
        d = self.root / token / role
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        (d / name).write_bytes(data)
        return {"token": token, "role": role, "name": name}

    def _files(self, token: str):
        if not re.fullmatch(r"[0-9a-f]{32}", token or ""):
            raise ValueError("unknown upload")
        files = {}
        for role in ("raw", "annotation"):
            found = list((self.root / token / role).glob("*")) if (self.root / token / role).is_dir() else []
            if not found:
                raise ValueError(f"the {'raw image' if role == 'raw' else 'annotated image'} is missing")
            files[role] = found[0]
        return files

    def check(self, token: str) -> dict:
        """Count the enclosed shapes per pod and prepare a preview."""
        f = self._files(token)
        cfg = self.session.cfg
        raw = load_image(f["raw"])
        try:
            labels, warns = extract_outlines(raw, load_image(f["annotation"]), cfg)
        except ValueError as exc:
            raise ValueError(str(exc))
        img = process_image(f["raw"], cfg)
        if img.error:
            raise ValueError(img.error)
        ev = evaluate_image(img, labels, None, cfg)
        n_shapes = int(labels.max())
        inside = sum(r["outline_count"] for r in ev["pods"])
        pods = [{"pod": r["pod"], "shapes": r["outline_count"],
                 "x": float(min(p[0] for p in img.pod(r["pod"]).outline)),
                 "y": float(np.mean([p[1] for p in img.pod(r["pod"]).bud_cut]))} for r in ev["pods"]]
        # preview: raw image, recognised outlines in magenta, pods numbered
        scale = min(1.0, 1600 / raw.shape[1])
        vis = raw.copy()
        vis[find_boundaries(labels, mode="thick")] = (255, 0, 255)
        for pod in img.pods:
            cv2.polylines(vis, [np.round(pod.outline).astype(np.int32).reshape(-1, 1, 2)], True, (40, 40, 40), 3, cv2.LINE_AA)
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in pods:
            x, y = int(p["x"] * scale), int(p["y"] * scale)
            label = f"Pod {p['pod']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = max(4, x - tw - 12)
            cv2.rectangle(vis, (tx - 4, y - th - 6), (tx + tw + 4, y + 6), (255, 255, 255), -1)
            cv2.putText(vis, label, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        (self.root / token / "preview.jpg").write_bytes(jpeg_bytes(vis, 85))
        known = {e.image.name for e in load_manifest(self.gt_dir)[0]}
        res = {"token": token, "raw": f["raw"].name, "annotation": f["annotation"].name, "pods": pods,
               "shapes_total": n_shapes, "shapes_outside": n_shapes - inside,
               "already_annotated": f["raw"].name in known, "warnings": warns + ev["warnings"]}
        self.checked[token] = res
        return res

    def preview(self, token: str) -> bytes:
        p = self.root / token / "preview.jpg"
        if not re.fullmatch(r"[0-9a-f]{32}", token or "") or not p.exists():
            raise ValueError("no preview")
        return p.read_bytes()

    def save(self, token: str, pod_counts) -> dict:
        res = self.checked.get(token)
        if not res:
            raise ValueError("check the annotation first")
        if not isinstance(pod_counts, list) or len(pod_counts) != len(res["pods"]):
            raise ValueError(f"type a seed count for each of the {len(res['pods'])} pods")
        try:
            counts = [int(c) for c in pod_counts]
        except (TypeError, ValueError):
            raise ValueError("seed counts must be whole numbers")
        if any(c < 0 for c in counts):
            raise ValueError("seed counts cannot be negative")
        f = self._files(token)
        gt = self.gt_dir
        raw_dir, ann_dir = gt / "raw", gt / "annotations"
        raw_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        if res["raw"] in {e.image.name for e in load_manifest(gt)[0]}:
            raise ValueError(f"{res['raw']} is already annotated; remove the old annotation first")
        raw_target = raw_dir / res["raw"]
        data = f["raw"].read_bytes()
        if raw_target.exists() and not _same_file(raw_target, data):
            raise ValueError(f"a different {res['raw']} is already in ground_truth/raw; rename the file first")
        ann_name = Path(res["raw"]).stem + "_annotation" + f["annotation"].suffix.lower()
        n = 2
        while (ann_dir / ann_name).exists():
            ann_name = f"{Path(res['raw']).stem}_annotation_{n}{f['annotation'].suffix.lower()}"
            n += 1
        shutil.copy(f["raw"], raw_target)
        shutil.copy(f["annotation"], ann_dir / ann_name)
        add_manifest_entry(gt / "manifest.yaml", res["raw"], ann_name, counts)
        shutil.rmtree(self.root / token, ignore_errors=True)
        self.checked.pop(token, None)
        diff = [p["pod"] for p, c in zip(res["pods"], counts) if p["shapes"] != c]
        return {"image": res["raw"], "annotation": ann_name, "pod_counts": counts, "mismatched_pods": diff}

    def cancel(self, token: str) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", token or ""):
            shutil.rmtree(self.root / token, ignore_errors=True)
            self.checked.pop(token, None)

    def listing(self) -> dict:
        try:
            entries, warnings = load_manifest(self.gt_dir)
        except ValueError as exc:
            return {"entries": [], "warnings": [str(exc)]}
        return {"entries": [{"image": e.image.name, "annotation": e.annotation.name if e.annotation else None,
                             "pod_counts": e.pod_counts} for e in entries], "warnings": warnings}

    def remove(self, image: str) -> dict:
        removed = remove_manifest_entry(self.gt_dir / "manifest.yaml", image)
        if not removed:
            raise ValueError(f"{image} is not in manifest.yaml")
        # keep the files, out of the way, in case the removal was a mistake
        bin_dir = self.gt_dir / "removed"
        bin_dir.mkdir(exist_ok=True)
        for sub, name in (("raw", image), ("annotations", removed.get("annotation"))):
            if name and (self.gt_dir / sub / name).exists():
                shutil.move(str(self.gt_dir / sub / name), str(bin_dir / name))
        return {"removed": image, "moved_to": str(bin_dir)}


# ------------------------------------------------------------------ manifest text edits (comments kept)
def add_manifest_entry(path: Path, image: str, annotation: str, counts: list[int]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# Ground-truth annotations. See MANUAL.md, sections 13 and 21.\n"
    if text and not text.endswith("\n"):
        text += "\n"
    entry = {"image": image, "annotation": annotation, "pod_counts": counts}
    block = (f"- image: {json.dumps(image, ensure_ascii=False)}\n"
             f"  annotation: {json.dumps(annotation, ensure_ascii=False)}\n"
             f"  pod_counts: [{', '.join(map(str, counts))}]\n")
    new = text + block
    data = yaml.safe_load(new)
    if not isinstance(data, list) or data[-1] != entry:
        raise ValueError("manifest.yaml could not be updated safely; please add the entry by hand")
    path.write_text(new, encoding="utf-8")


def remove_manifest_entry(path: Path, image: str) -> dict | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    match = next((d for d in data if isinstance(d, dict) and str(d.get("image")) == image), None)
    if match is None:
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, skip = [], False
    for line in lines:
        if line.startswith("- "):
            item = yaml.safe_load(line[2:]) if ":" in line else None
            skip = isinstance(item, dict) and str(item.get("image")) == image
        elif not line.startswith((" ", "\t")) and line.strip():
            skip = False
        if not skip:
            out.append(line)
    new = "".join(out)
    rest = yaml.safe_load(new) or []
    if len(rest) != len(data) - 1:
        raise ValueError("manifest.yaml could not be updated safely; please remove the entry by hand")
    path.write_text(new, encoding="utf-8")
    return match
