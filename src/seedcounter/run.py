"""Processing a whole input folder and writing all outputs."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import cv2

from . import __version__
from .io import jpeg_bytes, list_images, load_image, save_png
from .model import RunResult
from .pipeline import process_image
from .render import render
from .report import summaries, write_html, write_json, write_tables

log = logging.getLogger(__name__)


def analyse_folder(input_dir, cfg: dict, config_file: str = "config/default.yaml", progress=None) -> RunResult:
    """Automatic analysis of every image in a folder (no files written)."""
    input_dir = Path(input_dir)
    images = list_images(input_dir, cfg["input"]["extensions"], cfg["input"]["recursive"])
    if not images:
        raise FileNotFoundError(f"No {'/'.join(cfg['input']['extensions'])} files in {input_dir}")
    run = RunResult(date=dt.datetime.now().isoformat(timespec="seconds"), version=__version__,
                    config_file=str(config_file), config=cfg, input_dir=str(input_dir))
    for path in (progress(images) if progress else images):
        res = process_image(path, cfg)
        res.image = image_key(path, input_dir)
        run.images.append(res)
    return run


def image_key(path: Path, input_dir: Path) -> str:
    """File name, or sub-folder/file name for recursive runs."""
    path, input_dir = Path(path), Path(input_dir)
    if input_dir in path.parents and path.parent != input_dir:
        return str(path.relative_to(input_dir)).replace("\\", "/")
    return path.name


def write_outputs(run: RunResult, output_dir, cfg: dict, progress=None) -> list[dict]:
    """Annotated PNGs, tables, results.json and report.html for a run (automatic or reviewed)."""
    output_dir = Path(output_dir)
    out = cfg["output"]
    img_dir = output_dir / "annotated_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    thumbs: dict[str, bytes] = {}
    for res in (progress(run.images) if progress else run.images):
        if res.error or not (out.get("annotated_png", True) or out.get("report_html", True)):
            continue
        ann = render(res, load_image(res.path), cfg)
        if out.get("annotated_png", True):
            name = res.image.replace("/", "__").rsplit(".", 1)[0] + "_annotated.png"
            save_png(img_dir / name, ann)
            res.annotated_image = f"annotated_images/{name}"
        tw = cfg["render"]["thumbnail_width"]
        thumbs[res.image] = jpeg_bytes(cv2.resize(ann, (tw, int(ann.shape[0] * tw / ann.shape[1])), interpolation=cv2.INTER_AREA))
    sums = summaries(run, cfg)
    write_tables(run, cfg, output_dir / "output_table", sums)
    write_json(run, cfg, output_dir / "results.json", sums)
    if out.get("report_html", True):
        write_html(run, cfg, output_dir / "report.html", sums, thumbs)
    return sums


def run_folder(input_dir, output_dir, cfg: dict, config_file: str = "config/default.yaml", progress=None) -> tuple[RunResult, list[dict]]:
    run = analyse_folder(input_dir, cfg, config_file, progress)
    return run, write_outputs(run, output_dir, cfg)
