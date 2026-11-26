"""Loads and validates config/config.json."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.json"
# colors handed out to user-typed (open-vocabulary) objects
PROMPT_COLORS = ("#7CFFCB", "#FFD23D", "#FF7AE0", "#8AB4FF", "#FF9F6B", "#C6FF3D", "#6BE4FF", "#FF6B8A")


def hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


@dataclass
class ObjectSpec:
    coco: str
    label: str
    icon: str
    color: str
    enabled: bool = True
    class_id: int = -1  # filled in once the model's class names are known


@dataclass
class Config:
    preset: str
    weights: Path
    model_label: str
    device: str = "mps"
    imgsz: int = 640
    half: bool = True
    confidence: float = 0.35
    tracker: str = "bytetrack.yaml"
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 60
    mirror: bool = True
    objects: list[ObjectSpec] = field(default_factory=list)
    layers: dict[str, dict] = field(default_factory=dict)  # segment / pose / depth / yoloe
    presets: dict[str, dict] = field(default_factory=dict)  # light / medium / heavy -> weights, label
    mission_rounds: int = 5
    mission_seconds: float = 15.0
    mission_objects: list[str] = field(default_factory=list)  # names from `objects`; empty = the ticked ones
    mission_class_ids: list[int] = field(default_factory=list)

    def bind_class_ids(self, names: dict[int, str]) -> None:
        """Map each configured COCO name to the model's class id."""
        lookup = {name.lower(): idx for idx, name in names.items()}
        missing = [o.coco for o in self.objects if o.coco.lower() not in lookup]
        if missing:
            raise ValueError(f"Unknown class name(s) in config: {', '.join(missing)}")
        for obj in self.objects:
            obj.class_id = lookup[obj.coco.lower()]
        by_name = {}
        for obj in self.objects:
            by_name[obj.coco.lower()] = obj.class_id
            by_name[obj.label.lower()] = obj.class_id
        unknown = [n for n in self.mission_objects if n.lower() not in by_name]
        if unknown:
            raise ValueError(f"mission.objects must be names from 'objects', unknown: {', '.join(unknown)}")
        self.mission_class_ids = list(dict.fromkeys(by_name[n.lower()] for n in self.mission_objects))


def weights_label(path: Path) -> str:
    """'yolo26s-seg.pt' -> 'YOLO26-S-seg'."""
    m = re.match(r"(yolo\D*?)(\d+)([nsmlx])(.*)", path.stem, re.IGNORECASE)
    return f"{m[1].upper()}{m[2]}-{m[3].upper()}{m[4]}" if m else path.stem


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    presets = raw["model_presets"]
    preset = raw.get("model", "medium")
    if preset not in presets:
        raise ValueError(f"'model' must be one of {list(presets)}, got '{preset}'")

    all_presets = {}
    for name, spec in presets.items():
        w = _resolve(spec["weights"])
        all_presets[name] = {"weights": w, "label": spec.get("label") or weights_label(w)}
    weights = all_presets[preset]["weights"]
    label = all_presets[preset]["label"]

    layers = {}
    for name, spec in raw.get("layers", {}).items():
        w = _resolve(spec["weights"])
        layers[name] = {**spec, "weights": w, "label": spec.get("label") or weights_label(w)}
    missing = {"segment", "pose", "depth", "yoloe"} - set(layers)
    if missing:
        raise ValueError(f"config is missing layer weights for: {', '.join(sorted(missing))}")

    mission = raw.get("mission", {})
    rounds = int(mission.get("rounds", 5))
    seconds = float(mission.get("seconds_per_round", 15))
    if rounds < 1 or seconds < 3:
        raise ValueError("mission.rounds must be at least 1 and mission.seconds_per_round at least 3")
    inf = raw.get("inference", {})
    cam = raw.get("camera", {})
    objects = [ObjectSpec(**o) for o in raw.get("objects", [])]
    if not objects:
        raise ValueError("config needs at least one entry in 'objects'")

    return Config(
        preset=preset,
        weights=weights,
        model_label=label,
        device=inf.get("device", "mps"),
        imgsz=int(inf.get("imgsz", 640)),
        half=bool(inf.get("half", True)),
        confidence=float(inf.get("confidence", 0.35)),
        tracker=inf.get("tracker", "bytetrack.yaml"),
        camera_index=int(cam.get("index", 0)),
        camera_width=int(cam.get("width", 1280)),
        camera_height=int(cam.get("height", 720)),
        camera_fps=int(cam.get("fps", 60)),
        mirror=bool(cam.get("mirror", True)),
        objects=objects,
        layers=layers,
        presets=all_presets,
        mission_rounds=rounds,
        mission_seconds=seconds,
        mission_objects=[str(n) for n in mission.get("objects", [])],
    )
