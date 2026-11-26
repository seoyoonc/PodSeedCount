"""Result data model.

Design rule (for the planned interactive review version): only *observations* are stored
(seed outlines, seed features and confidences, pod geometry, evidence). Everything a reviewer can
change the outcome of - seed colour classes, pod counts, pod confidence, needs_review - is derived
on demand by ``seedcounter.summary`` from the stored observations and the current thresholds.
Edits (``seedcounter.edits``) change observations; summaries are then simply recomputed.
All coordinates are in full-resolution image pixels, (x, y).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

Point = list  # [x, y]


@dataclass
class Seed:
    id: str                              # "<pod>-<n>", stable across edits
    pod: int
    contour: list                        # [[x, y], ...]
    centroid: list                       # [x, y]
    area_px: float
    major_px: float
    minor_px: float
    row: str                             # "top" | "bottom" (relative to the centerline)
    confidence: float                    # 0..1 from the automatic model
    features: dict = field(default_factory=dict)
    source: str = "auto"                 # "auto" | "manual"
    status: str = "active"               # "active" | "rejected"
    reviewed: Optional[str] = None       # None | "accepted" | "split" (set by a reviewer)


@dataclass
class Pod:
    index: int                           # 1-based, top to bottom
    outline: list                        # polygon of the pod [[x, y], ...]
    centerline: list                     # polyline [[x, y], ...]
    bud_cut: list                        # line [[x, y], [x, y]] - nothing left of it is counted
    stem_cut: list                       # line [[x, y], [x, y]] - nothing right of it is counted
    body_width_px: float
    bud_visible: bool = True
    stem_visible: bool = True
    uncovered: list = field(default_factory=list)   # dark seed-like areas without a seed: {contour, area_px[, dismissed]}
    warnings: list = field(default_factory=list)
    seeds: list = field(default_factory=list)       # list[Seed]
    status: str = "active"               # "active" | "ignored" (not a pod, set by a reviewer)
    manual_flag: Optional[str] = None    # None (automatic) | "no" | "maybe" | "yes": review flag set by a reviewer

    @property
    def bud_x(self) -> float:
        return float(self.bud_cut[0][0])

    @property
    def stem_x(self) -> float:
        return float(self.stem_cut[0][0])

    def active_seeds(self) -> list:
        """Counted seeds: not rejected, and between the bud and stem cuts."""
        if self.status != "active":
            return []
        return [s for s in self.seeds if s.status == "active" and self.bud_x < s.centroid[0] < self.stem_x]

    def mean_y(self) -> float:
        return float(sum(p[1] for p in self.centerline) / max(len(self.centerline), 1))


@dataclass
class ImageResult:
    image: str                           # file name
    path: str
    width: int
    height: int
    seed_radius_px: float
    pods: list = field(default_factory=list)        # list[Pod]
    warnings: list = field(default_factory=list)
    error: Optional[str] = None                     # set when the image could not be processed
    annotated_image: Optional[str] = None           # relative path of the PNG
    reviewed: bool = False                          # a person has checked the whole image
    manual_flag: Optional[str] = None               # None (automatic) | "no" | "maybe" | "yes": set by a reviewer

    def find_seed(self, seed_id: str):
        for pod in self.pods:
            for s in pod.seeds:
                if s.id == seed_id:
                    return pod, s
        raise KeyError(f"No seed {seed_id} in {self.image}")

    def new_seed_id(self, pod_index: int) -> str:
        """An id "<pod>-<n>" not used anywhere in the image (seeds can move between pods)."""
        used = {s.id for p in self.pods for s in p.seeds}
        n = 1 + max((int(i.split("-")[-1]) for i in used if i.startswith(f"{pod_index}-") and i.split("-")[-1].isdigit()), default=0)
        while f"{pod_index}-{n}" in used:
            n += 1
        return f"{pod_index}-{n}"

    def active_pods(self) -> list:
        """Pods that count, top to bottom."""
        return sorted((p for p in self.pods if p.status == "active"), key=lambda p: p.mean_y())

    def pod(self, index: int) -> Pod:
        for p in self.pods:
            if p.index == index:
                return p
        raise KeyError(f"No pod {index} in {self.image}")


@dataclass
class RunResult:
    date: str
    version: str
    config_file: str
    config: dict
    input_dir: str
    images: list = field(default_factory=list)      # list[ImageResult]


# ---------------------------------------------------------------- serialisation
def to_dict(obj: Any) -> dict:
    return asdict(obj)


def seed_from_dict(d: dict) -> Seed:
    return Seed(**d)


def pod_from_dict(d: dict) -> Pod:
    d = dict(d)
    d["seeds"] = [seed_from_dict(s) for s in d.get("seeds", [])]
    return Pod(**d)


def image_from_dict(d: dict) -> ImageResult:
    d = dict(d)
    d["pods"] = [pod_from_dict(p) for p in d.get("pods", [])]
    return ImageResult(**d)


def run_from_dict(d: dict) -> RunResult:
    d = dict(d)
    d["images"] = [image_from_dict(i) for i in d.get("images", [])]
    return RunResult(**d)
