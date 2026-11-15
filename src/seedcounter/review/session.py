"""Review state: automatic results + a reviewer's edit log per image, saved after every change.

Files (inside the output folder):
  review/cache/<image>.json     automatic result, reused while the image and settings are unchanged
  review/sessions/<image>.json  the analysis the reviewer worked on ("base") + the edit log
Results shown and exported are always base + edits replayed, so undo is simply "drop the last edit".
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
import threading
from pathlib import Path

import cv2

from .. import ANALYSIS_VERSION, __version__
from ..edits import apply_edits
from ..io import jpeg_bytes, list_images, load_image
from ..model import ImageResult, RunResult, image_from_dict, to_dict
from ..pipeline import process_image
from ..run import image_key, write_outputs
from ..summary import image_summary, pod_summary, seed_class

log = logging.getLogger(__name__)
ANALYSIS_KEYS = ("processing", "pods", "contrast", "segmentation")


def analysis_hash(cfg: dict) -> str:
    """Changes only when settings that affect the automatic analysis change (not thresholds)."""
    part = {k: cfg[k] for k in ANALYSIS_KEYS}
    part["seed_model"] = cfg["confidence"]["seed_model"]
    part["version"] = ANALYSIS_VERSION
    return hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=float), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written file


def reviewed_ground_truth(output_dir) -> tuple[list, list[str]]:
    """Images marked as checked in the review, as ground truth: [(GTEntry, labels, borders), ...].

    labels: the final seeds (automatic ones not rejected + added ones, inside the cuts of real pods)
    drawn into a label image, exactly like hand-drawn outlines; used for counts and detection.
    borders: only the seeds whose border a person explicitly redrew; used to judge border fit
    (None when the image has no redrawn seeds). Accepted seeds confirm detection, not shape.
    """
    import numpy as np
    from ..ground_truth import GTEntry
    out, warnings = [], []
    for f in sorted((Path(output_dir) / "review" / "sessions").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            img = apply_edits(image_from_dict(d["base"]), d.get("edits", []))
        except Exception as exc:
            warnings.append(f"{f.name}: review file could not be read ({exc})")
            continue
        if not img.reviewed:
            continue
        path = Path(img.path)
        if not path.exists():
            warnings.append(f"{img.image}: checked in the review, but {path} no longer exists; skipped")
            continue
        labels = np.zeros((img.height, img.width), np.int32)
        borders = np.zeros((img.height, img.width), np.int32)
        n = m = 0
        for pod in img.active_pods():
            for s in pod.active_seeds():
                n += 1
                poly = [np.round(np.asarray(s.contour)).astype(np.int32)]
                cv2.fillPoly(labels, poly, n)
                if s.features.get("redrawn"):
                    m += 1
                    cv2.fillPoly(borders, poly, m)
        out.append((GTEntry(path, None, None), labels, borders if m else None))
    return out, warnings


class ReviewSession:
    def __init__(self, input_dir, output_dir, cfg: dict, config_file: str = "", gt_dir=None):
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.job = {"kind": None, "state": "idle", "done": 0, "total": 0, "message": "", "result": None}
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.config_file = str(config_file)
        self.hash = analysis_hash(cfg)
        self.cache_dir = self.output_dir / "review" / "cache"
        self.session_dir = self.output_dir / "review" / "sessions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # An empty or missing input folder is fine: the page opens with an empty list, ready for "Add images".
        self.input_dir.mkdir(parents=True, exist_ok=True)
        paths = list_images(self.input_dir, cfg["input"]["extensions"], cfg["input"]["recursive"])
        self.paths = {image_key(p, self.input_dir): p for p in paths}
        self.keys = list(self.paths)
        self.base: dict[str, ImageResult] = {}
        self.edits: dict[str, list] = {}
        self.base_hash: dict[str, str] = {}
        self.current: dict[str, ImageResult] = {}
        self.lock = threading.RLock()
        self._jpeg: dict[str, bytes] = {}

    # ------------------------------------------------------------ loading
    def load_all(self, progress=None) -> None:
        for key in (progress(self.keys) if progress else self.keys):
            self._load(key)

    def _auto_result(self, key: str) -> ImageResult:
        path = self.paths[key]
        st = path.stat()
        cache = self.cache_dir / (_safe(key) + ".json")
        if cache.exists():
            try:
                d = json.loads(cache.read_text(encoding="utf-8"))
                if d["hash"] == self.hash and d["size"] == st.st_size and d["mtime"] == st.st_mtime:
                    return image_from_dict(d["result"])
            except Exception:  # unreadable cache: just re-analyse
                pass
        res = process_image(path, self.cfg)
        res.image = key
        _write_json(cache, {"hash": self.hash, "size": st.st_size, "mtime": st.st_mtime, "result": to_dict(res)})
        return res

    def _load(self, key: str) -> None:
        sess = self.session_dir / (_safe(key) + ".json")
        if sess.exists():
            try:
                d = json.loads(sess.read_text(encoding="utf-8"))
                self.base[key] = image_from_dict(d["base"])
                self.edits[key] = list(d.get("edits", []))
                self.base_hash[key] = d.get("config_hash", "")
                self.current[key] = self._replay(key)
                return
            except Exception as exc:
                log.warning("Review file for %s could not be read (%s); starting fresh", key, exc)
        self.base[key] = self._auto_result(key)
        self.edits[key] = []
        self.base_hash[key] = self.hash
        self.current[key] = copy.deepcopy(self.base[key])

    def _replay(self, key: str, edits=None) -> ImageResult:
        img = copy.deepcopy(self.base[key])
        good = []
        for e in self.edits[key] if edits is None else edits:
            try:
                apply_edits(img, [e])
                good.append(e)
            except (KeyError, ValueError) as exc:  # e.g. edited by hand; skip what no longer applies
                log.warning("%s: skipped edit %s (%s)", key, e, exc)
        if edits is None:
            self.edits[key] = good
        return img

    def _save(self, key: str) -> None:
        sess = self.session_dir / (_safe(key) + ".json")
        if not self.edits[key]:
            sess.unlink(missing_ok=True)
            return
        _write_json(sess, {"version": 1, "image": key, "config_hash": self.base_hash[key],
                           "updated": dt.datetime.now().isoformat(timespec="seconds"),
                           "edits": self.edits[key], "base": to_dict(self.base[key])})

    # ------------------------------------------------------------ changes
    def apply(self, key: str, edit: dict) -> None:
        with self.lock:
            trial = copy.deepcopy(self.current[key])
            apply_edits(trial, [edit])          # raises ValueError / KeyError for invalid edits
            self.edits[key].append(edit)
            self.current[key] = trial
            self._save(key)

    def undo(self, key: str) -> bool:
        with self.lock:
            if not self.edits[key]:
                return False
            self.edits[key].pop()
            self.current[key] = self._replay(key)
            self._save(key)
            return True

    def discard(self, key: str) -> None:
        """Drop all edits and re-analyse with the current settings."""
        with self.lock:
            self.edits[key] = []
            self._save(key)
            self.base[key] = self._auto_result(key)
            self.base_hash[key] = self.hash
            self.current[key] = copy.deepcopy(self.base[key])

    # ------------------------------------------------------------ views
    def config_with(self, review=None, reject=None) -> dict:
        cfg = copy.deepcopy(self.cfg)
        c = cfg["confidence"]
        if review is not None:
            c["review_threshold"] = min(max(float(review), 0.0), 1.0)
        if reject is not None:
            c["reject_threshold"] = min(max(float(reject), 0.0), c["review_threshold"])
        return cfg

    def listing(self, cfg: dict) -> list[dict]:
        out = []
        for i, key in enumerate(self.keys):
            img = self.current[key]
            s = image_summary(img, cfg)
            out.append({"id": i, "key": key, "seeds": s["seeds"], "pods": s["pods"], "needs_review": s["needs_review"],
                        "reviewed": img.reviewed, "edits": len(self.edits[key]), "auto_seeds": s["auto_seeds"],
                        "change": s["change"], "abs_change": s["abs_change"], "green": s["green"],
                        "orange": s["orange"], "red": s["red"], "pods_flagged": s["pods_maybe"] + s["pods_yes"],
                        "redrawn": sum(bool(sd.features.get("redrawn")) for p in img.active_pods() for sd in p.active_seeds()),
                        "note": s["note"]})
        return out

    def totals(self, listing: list[dict]) -> dict:
        checked = [x for x in listing if x["reviewed"]]
        seeds = sum(x["seeds"] for x in checked)
        pods = sum(x["pods"] for x in checked)
        off = sum(x["abs_change"] for x in checked)
        return {"images": len(listing), "checked": len(checked), "edited": sum(x["edits"] > 0 for x in listing),
                "need_review": sum(x["needs_review"] != "no" and not x["reviewed"] for x in listing),
                "checked_pods": pods, "checked_seeds": seeds, "checked_abs_change": off,
                "checked_rel_change": off / seeds if seeds else None}

    def payload(self, i: int, cfg: dict) -> dict:
        key = self.keys[i]
        img = self.current[key]
        s = image_summary(img, cfg)
        notes = [s["note"]] if s["note"] else []
        if self.base_hash[key] != self.hash and self.edits[key]:
            notes.append("This image was reviewed on an analysis made with earlier settings. "
                         "'Discard edits' re-analyses it with the current settings.")
        pods = []
        order = img.active_pods() + [p for p in img.pods if p.status != "active"]
        for pod in order:
            counted = {id(sd) for sd in pod.active_seeds()}
            pods.append({
                "index": pod.index, "status": pod.status, "outline": pod.outline, "centerline": pod.centerline,
                "bud_cut": pod.bud_cut, "stem_cut": pod.stem_cut,
                "summary": pod_summary(pod, cfg, img.seed_radius_px, img.reviewed) if pod.status == "active" else None,
                "uncovered": [{"area": j, "contour": u["contour"], "area_px": u["area_px"],
                               "dismissed": bool(u.get("dismissed"))} for j, u in enumerate(pod.uncovered)],
                "seeds": [{"id": sd.id, "contour": sd.contour, "centroid": sd.centroid, "confidence": sd.confidence,
                           "cls": ("split" if sd.reviewed == "split" else "rejected" if sd.status != "active"
                                   else seed_class(sd, cfg) if id(sd) in counted else "outside"),
                           "status": sd.status, "source": sd.source, "reviewed": sd.reviewed, "row": sd.row,
                           "area_px": sd.area_px, "features": sd.features} for sd in pod.seeds],
            })
        return {"id": i, "key": key, "width": img.width, "height": img.height, "reviewed": img.reviewed,
                "error": img.error, "notes": notes, "edits": len(self.edits[key]),
                "summary": {k: v for k, v in s.items() if k != "pod_rows"}, "pods": pods,
                "thresholds": {"review": cfg["confidence"]["review_threshold"],
                               "reject": cfg["confidence"]["reject_threshold"]}}

    def picture(self, i: int) -> bytes:
        key = self.keys[i]
        if key not in self._jpeg:
            if len(self._jpeg) >= 4:
                self._jpeg.pop(next(iter(self._jpeg)))
            self._jpeg[key] = jpeg_bytes(load_image(self.paths[key]), quality=90)
        return self._jpeg[key]

    # ------------------------------------------------------------ background jobs
    def _run_job(self, kind: str, fn) -> dict:
        with self.lock:
            if self.job["state"] == "running":
                raise ValueError("another task is still running")
            self.job = {"kind": kind, "state": "running", "done": 0, "total": 0, "message": "starting", "result": None}

        def progress(items):
            items = list(items)
            self.job["total"] = len(items)
            for k, it in enumerate(items):
                self.job["done"] = k
                yield it
            self.job["done"] = len(items)

        def work():
            try:
                self.job["result"] = fn(progress)
                self.job["state"] = "done"
            except ValueError as exc:  # an expected situation (e.g. nothing checked yet): no traceback
                log.info("%s: %s", kind, exc)
                self.job["state"], self.job["message"] = "error", str(exc)
            except Exception as exc:
                log.exception("%s failed", kind)
                self.job["state"], self.job["message"] = "error", str(exc)

        threading.Thread(target=work, daemon=True).start()
        return self.job

    def start_calibration(self, use_gt: bool = True, sweep_grid: bool = True) -> dict:
        from ..calibrate import GTSet, calibrate

        def fn(progress):
            self.job["message"] = "collecting ground truth"
            gt = GTSet(self.gt_dir if use_gt else None, self.cfg)
            extra, warns = reviewed_ground_truth(self.output_dir)
            gt.warnings += warns
            gt.add(extra)
            if not extra and not gt.entries:
                raise ValueError("no checked images yet: mark images as checked first")
            self.job["message"] = f"testing settings on {len(gt.entries)} images ({len(extra)} from the review)"
            res = calibrate(gt, self.cfg, sweep_grid, progress if sweep_grid else None)
            return {"before": res["before"], "after": res["after"], "changes": res["changes"],
                    "images": res["images"], "reviewed_images": len(extra), "warnings": res["warnings"],
                    "redrawn_seeds": res["after"].get("border_seeds", 0)}
        return self._run_job("calibrate", fn)

    def start_apply_calibration(self) -> dict:
        from ..calibrate import apply_to_file
        from ..config import load_config
        last = self.job.get("result") if self.job.get("kind") == "calibrate" else None
        if not last or self.job["state"] != "done":
            raise ValueError("run 'Learn from reviews' first")
        if not self.config_file or not Path(self.config_file).exists():
            raise ValueError("the settings file is unknown; use scripts/calibrate.py --from-reviews --apply")
        changes = last["changes"]

        def fn(progress):
            backup = apply_to_file(self.config_file, changes)
            cfg = load_config(self.config_file)
            with self.lock:
                self.cfg, self.hash = cfg, analysis_hash(cfg)
            self.job["message"] = "analysing unedited images with the new settings"
            redo = [k for k in self.keys if not self.edits[k]]
            for k in progress(redo):
                res = self._auto_result(k)
                with self.lock:
                    if not self.edits[k]:
                        self.base[k], self.base_hash[k], self.current[k] = res, self.hash, copy.deepcopy(res)
            return {"applied": len(changes), "backup": str(backup), "reanalysed": len(redo)}
        return self._run_job("apply", fn)

    def export(self, cfg: dict) -> dict:
        """Write the normal outputs (PNG, CSV, HTML, JSON) with all edits applied."""
        if not self.keys:
            raise ValueError("there are no images yet: add images first")
        with self.lock:
            images = [copy.deepcopy(self.current[k]) for k in self.keys]
        for k, img in zip(self.keys, images):
            img.path = str(self.paths[k])
        run = RunResult(date=dt.datetime.now().isoformat(timespec="seconds"), version=__version__,
                        config_file=self.config_file, config=cfg, input_dir=str(self.input_dir), images=images)
        sums = write_outputs(run, self.output_dir, cfg)
        return {"images": len(sums), "reviewed": sum(s["reviewed"] == "yes" for s in sums),
                "need_review": sum(s["needs_review"] != "no" for s in sums),
                "report": str(self.output_dir / "report.html")}
