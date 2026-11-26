"""Inference worker: camera frame -> YOLO26 engines -> display-ready packet.

Engines (one runs per frame, chosen live from the UI):
  detect    YOLO26 detector + ByteTrack                (COCO classes)
  segment   YOLO26-seg + ByteTrack                     (COCO classes, with masks)
  yoloe     YOLOE-26 open-vocabulary + ByteTrack       (any words the user types)
Optional layers on top:
  pose      YOLO26-pose skeletons (also powers face privacy)
  depth     YOLO26-depth metric depth: distance per object, depth view, portrait blur
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .camera import CameraStream
from .config import PROMPT_COLORS, Config, hex_to_bgr

CUSTOM_BASE = 1000  # class keys for user-typed (open vocabulary) objects start here
PROMPT_CONF_SCALE = 0.7  # YOLOE's open-vocabulary scores run lower than YOLO26's, so typed words use 70% of the slider

# COCO-17 skeleton grouped by limb, each limb gets its own neon hue (BGR)
SKELETON = (
    (((5, 7), (7, 9)), (255, 232, 61)),            # left arm   cyan
    (((6, 8), (8, 10)), (154, 61, 255)),           # right arm  pink
    (((11, 13), (13, 15)), (138, 255, 61)),        # left leg   green
    (((12, 14), (14, 16)), (61, 194, 255)),        # right leg  amber
    (((5, 6), (5, 11), (6, 12), (11, 12)), (255, 107, 157)),  # torso violet
    (((0, 1), (0, 2), (1, 3), (2, 4)), (255, 255, 255)),      # face
)
KP_CONF = 0.45


@dataclass
class Detection:
    tid: int  # tracker id, or -1 while the tracker has not confirmed it
    cls: int  # COCO class id, or CUSTOM_BASE + i for typed prompts
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    poly: np.ndarray | None = None  # mask outline (N, 2) in source pixels
    dist: float | None = None  # meters, when the depth layer is on


@dataclass
class Packet:
    image: np.ndarray  # RGB, resized for the view (physical pixels)
    scale: float  # source pixel -> view logical pixel
    disp_w: int
    disp_h: int
    src_w: int
    src_h: int
    detections: list[Detection]
    faces: list[tuple[int, int, int, int]]
    depth_range: tuple[float, float] | None
    engine: str
    engine_label: str
    infer_ms: float
    cam_fps: float
    dpr: float
    stamp: float = field(default_factory=time.perf_counter)


class LiveSettings:
    """Values the UI thread writes and the worker reads (attribute swaps are atomic)."""

    def __init__(self, cfg: Config, source):
        self.class_ids: tuple[int, ...] = tuple(o.class_id for o in cfg.objects if o.enabled)
        self.prompts: tuple[str, ...] = ()
        self.preset = cfg.preset  # light / medium / heavy, switchable live from the panel
        self.mission_ids: tuple[int, ...] = ()  # mission targets are always detected, even if unticked
        self.confidence = cfg.confidence
        self.mirror = cfg.mirror
        self.privacy = False
        self.masks = False
        self.pose = False
        self.depth_view = False
        self.portrait = False
        self.view = (1280, 720, 2.0)  # logical w, h, device pixel ratio
        self.source = source
        self.source_gen = 0  # bump to make the worker reopen the source

    @property
    def detect_ids(self) -> tuple[int, ...]:
        return self.class_ids + tuple(c for c in self.mission_ids if c not in self.class_ids)

    @property
    def need_depth(self) -> bool:
        return self.depth_view or self.portrait


def faces_from_pose(poses: list[np.ndarray], w: int, h: int):
    """Face boxes from nose/eyes/ears keypoints — far more robust than a face cascade."""
    out = []
    for k in poses:
        head = k[:5]
        pts = head[head[:, 2] > 0.45][:, :2]
        if len(pts) < 2:
            continue
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        side = max(x1 - x0, y1 - y0) * 1.9
        sh = k[5:7]
        if (sh[:, 2] > 0.45).all():
            side = max(side, abs(sh[0, 0] - sh[1, 0]) * 0.55)
        side = max(side, 24)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - side * 0.08
        out.append((int(max(0, cx - side * 0.55)), int(max(0, cy - side * 0.7)),
                    int(min(w, cx + side * 0.55)), int(min(h, cy + side * 0.6))))
    return out


def pixelate(frame: np.ndarray, boxes):
    for x0, y0, x1, y1 in boxes:
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = frame[y0:y1, x0:x1]
        tiny = cv2.resize(roi, (max(1, (x1 - x0) // 14), max(1, (y1 - y0) // 14)), interpolation=cv2.INTER_AREA)
        frame[y0:y1, x0:x1] = cv2.resize(tiny, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


class DetectorWorker(QThread):
    packet_ready = Signal()
    status = Signal(str)
    model_ready = Signal(str)
    preset_changed = Signal(str, str)  # preset name, model label

    def __init__(self, cfg: Config, settings: LiveSettings, model):
        super().__init__()
        self.cfg = cfg
        self.settings = settings
        self.models = {"detect": model}
        self._detectors = {cfg.preset: model}  # all presets, so switching in the UI is instant
        self._preset = cfg.preset
        self._detect_label = cfg.model_label
        self._running = True
        self._lock = threading.Lock()
        self._packet: Packet | None = None
        self._yoloe_names: list[str] | None = None
        self._yoloe_keys: list[int] = []
        self._depth: np.ndarray | None = None
        self._poses: list[np.ndarray] | None = None
        # adaptive scheduler: estimated cost (ms) and last frame each extra layer ran on
        self._cost = {"pose": 10.0, "depth": 14.0}
        self._last_run = {"pose": 0, "depth": 0}
        self._near_far = None
        self._frame_n = 0

    def latest(self) -> Packet | None:
        with self._lock:
            pkt, self._packet = self._packet, None
        return pkt

    def stop(self):
        self._running = False
        self.wait(5000)

    # --------------------------------------------------------------- models
    def _kw(self):
        return dict(imgsz=self.cfg.imgsz, device=self.cfg.device,
                    quantize=16 if self.cfg.half else 32, verbose=False)

    def _load(self, name: str):
        if name in self.models:
            return self.models[name]
        from ultralytics import YOLO, YOLOE

        spec = self.cfg.layers[name]
        self.status.emit(f"Loading {spec['label']}…")
        path = spec["weights"]
        if name == "yoloe":
            # the text encoder is fetched next to the weights, never into the project root
            with contextlib.chdir(path.parent):
                model = YOLOE(str(path))
                model.set_classes([o.coco for o in self.cfg.objects])
        else:
            model = YOLO(str(path))
        dummy = np.zeros((self.cfg.camera_height, self.cfg.camera_width, 3), np.uint8)
        for _ in range(2):
            model.predict(dummy, **self._kw())
        if name == "segment":  # it tracks, so set its tracker up now rather than on the first toggle
            model.track(dummy, persist=True, tracker=self.cfg.tracker, **self._kw())
            self._reset_tracker(model)
        self.models[name] = model
        return model

    def _preload(self):
        self.status.emit(f"Warming up {self.cfg.model_label} on {self.cfg.device.upper()}…")
        dummy = np.zeros((self.cfg.camera_height, self.cfg.camera_width, 3), np.uint8)
        for _ in range(3):
            self.models["detect"].predict(dummy, **self._kw())
        for name in ("segment", "pose", "depth"):  # small; preloading keeps the live demo stall-free
            if not self._running:
                return
            self._load(name)
        for name in self.cfg.presets:  # the other detector sizes (6-40 MB each)
            if not self._running:
                return
            self._load_detector(name)
        self.model_ready.emit(f"{self.cfg.model_label} ready on {self.cfg.device.upper()}")

    def _load_detector(self, preset: str):
        if preset in self._detectors:
            return self._detectors[preset]
        from ultralytics import YOLO

        spec = self.cfg.presets[preset]
        self.status.emit(f"Loading {spec['label']}…")
        spec["weights"].parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(spec["weights"]))  # downloads the official weights on first use if missing
        if {i: n.lower() for i, n in model.names.items()} != \
                {i: n.lower() for i, n in self._detectors[self.cfg.preset].names.items()}:
            raise ValueError(f"{spec['label']} has different class names than {self.cfg.model_label}")
        dummy = np.zeros((self.cfg.camera_height, self.cfg.camera_width, 3), np.uint8)
        for _ in range(3):
            model.predict(dummy, **self._kw())
        model.track(dummy, persist=True, tracker=self.cfg.tracker, **self._kw())  # set the tracker up now
        self._reset_tracker(model)
        self._detectors[preset] = model
        return model

    def _switch_preset(self, preset: str):
        """Swap the main detector (light / medium / heavy) between two frames."""
        try:
            model = self._load_detector(preset)
        except Exception as exc:
            self.status.emit(f"Could not load the {preset} model: {exc}")
            self.settings.preset = self._preset
            self.preset_changed.emit(self._preset, self._detect_label)
            return
        self._reset_tracker(model)
        self.models["detect"] = model
        self._preset = preset
        self._detect_label = self.cfg.presets[preset]["label"]
        self.preset_changed.emit(preset, self._detect_label)

    def _ensure_yoloe_classes(self, model):
        s = self.settings
        by_id = {o.class_id: o.coco for o in self.cfg.objects}
        ids = s.detect_ids
        keys = [c for c in ids if c in by_id] + [CUSTOM_BASE + i for i in range(len(s.prompts))]
        names = [by_id[c] for c in ids if c in by_id] + list(s.prompts)
        if names != self._yoloe_names:
            with contextlib.chdir(self.cfg.layers["yoloe"]["weights"].parent):
                model.set_classes(names)
            self._yoloe_names, self._yoloe_keys = names, keys
            self._reset_tracker(model)

    # ------------------------------------------------------------------ run
    def run(self):
        try:
            self._preload()
        except Exception as exc:
            self.status.emit(f"Model failed to load: {exc}")
            return
        cam: CameraStream | None = None
        gen = -1
        last_id = -1
        last_error = None
        last_exc = None
        while self._running:
            s = self.settings
            if cam is None or gen != s.source_gen:
                if cam:
                    cam.stop()
                gen = s.source_gen
                cam = CameraStream(s.source, self.cfg.camera_width, self.cfg.camera_height, self.cfg.camera_fps)
                cam.start()
                cam.opened.wait(5.0)
                last_id = -1
                for m in self.models.values():
                    self._reset_tracker(m)
                if cam.error and isinstance(s.source, int) and s.source != self.cfg.camera_index:
                    self.status.emit(f"No camera at index {s.source}, switching back.")
                    s.source = self.cfg.camera_index
                    s.source_gen += 1
                    continue

            frame, fid = cam.wait_latest(last_id, 0.5)
            if frame is None:
                if cam.error and cam.error != last_error:
                    last_error = cam.error
                    self.status.emit(cam.error)
                continue
            last_id = fid
            if s.preset != self._preset:
                self._switch_preset(s.preset)
            if s.mirror and isinstance(s.source, int):
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            try:
                pkt = self._process(frame, cam.fps)
            except Exception as exc:  # keep the stream alive, surface the problem once
                if str(exc) != last_exc:
                    last_exc = str(exc)
                    self.status.emit(f"Inference error: {exc}")
                continue
            pkt.infer_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._packet = pkt
            self.packet_ready.emit()

        if cam:
            cam.stop()

    def _process(self, frame: np.ndarray, cam_fps: float) -> Packet:
        s = self.settings
        self._frame_n += 1
        h, w = frame.shape[:2]

        t_start = time.perf_counter()
        engine = "yoloe" if s.prompts else ("segment" if s.masks else "detect")
        model = self._load(engine)
        dets = self._detect(model, engine, frame)

        if s.pose or s.privacy:
            if self._poses is None or self._due("pose", t_start, cam_fps, max_stale=3):
                t = time.perf_counter()
                r = self.models["pose"].predict(frame, conf=0.4, **self._kw())[0]
                self._poses = list(r.keypoints.data.cpu().numpy()) if r.keypoints is not None else []
                self._ran("pose", t)
        else:
            self._poses = None
        poses = self._poses or []

        depth_range = None
        if s.need_depth:
            if self._depth is None or self._depth.shape != (h, w) or self._due("depth", t_start, cam_fps, max_stale=6):
                t = time.perf_counter()
                r = self.models["depth"].predict(frame, **self._kw())[0]
                d = r.depth.data
                self._depth = (d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)).astype(np.float32)
                self._ran("depth", t)
            depth = self._depth
            for d in dets:
                d.dist = self._box_distance(depth, d.box)
            near, far = np.percentile(depth[::8, ::8], (3, 97))
            if self._near_far is None:
                self._near_far = (near, far)
            else:  # smooth the colormap range so it doesn't flicker
                n0, f0 = self._near_far
                self._near_far = (n0 * 0.85 + near * 0.15, f0 * 0.85 + far * 0.15)
            depth_range = self._near_far
            if s.portrait:
                frame = self._portrait(frame, depth, dets)
            if s.depth_view:
                frame = self._depth_overlay(frame, depth)
        else:
            self._depth = None

        faces = faces_from_pose(poses, w, h) if s.privacy else []
        if s.masks:
            frame = self._draw_masks(frame, dets)
        if s.pose and poses:
            frame = self._draw_poses(frame, poses)
        if faces:
            pixelate(frame, faces)

        labels = {"detect": self._detect_label, "segment": self.cfg.layers["segment"]["label"],
                  "yoloe": self.cfg.layers["yoloe"]["label"]}
        return self._prepare(frame, dets, faces, depth_range, engine,
                             labels[engine], cam_fps)

    def _due(self, layer: str, t_start: float, cam_fps: float, max_stale: int) -> bool:
        """Run an extra layer only if it still fits in this frame's time budget.

        The detector runs on every frame; pose / depth fill the remaining time and otherwise reuse
        their last result (but never older than `max_stale` frames). This keeps the video at full
        camera rate even with the heavy preset and several layers stacked.
        """
        if self._frame_n - self._last_run[layer] >= max_stale:
            return True
        fps = min(60.0, max(20.0, cam_fps or 30.0))
        budget_ms = 1000.0 / fps * 0.9
        spent_ms = (time.perf_counter() - t_start) * 1000.0
        return spent_ms + self._cost[layer] + 5.0 <= budget_ms  # 5 ms for drawing + resize

    def _ran(self, layer: str, t0: float):
        self._cost[layer] = self._cost[layer] * 0.8 + (time.perf_counter() - t0) * 1000.0 * 0.2
        self._last_run[layer] = self._frame_n

    def _detect(self, model, engine: str, frame) -> list[Detection]:
        s = self.settings
        if engine == "yoloe":
            self._ensure_yoloe_classes(model)
            if not self._yoloe_names:
                return []
            classes = None
        else:
            classes = list(s.detect_ids)
            if not classes:
                return []
        conf = s.confidence * PROMPT_CONF_SCALE if engine == "yoloe" else s.confidence
        res = model.track(frame, persist=True, tracker=self.cfg.tracker, conf=conf,
                          classes=classes, **self._kw())[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(cls), -1)
        polys = [None] * len(cls)
        if s.masks and res.masks is not None:
            polys = [p.astype(np.int32) if len(p) > 2 else None for p in res.masks.xy]
        if engine == "yoloe":
            cls = np.array([self._yoloe_keys[c] if c < len(self._yoloe_keys) else -1 for c in cls])
        return [Detection(int(i), int(c), float(p), tuple(map(float, b)), poly)
                for b, p, c, i, poly in zip(xyxy, conf, cls, ids, polys) if c >= 0]

    def _color(self, key: int) -> tuple[int, int, int]:
        if key >= CUSTOM_BASE:
            return hex_to_bgr(PROMPT_COLORS[(key - CUSTOM_BASE) % len(PROMPT_COLORS)])
        obj = next((o for o in self.cfg.objects if o.class_id == key), None)
        return hex_to_bgr(obj.color) if obj else (255, 255, 255)

    def _draw_masks(self, frame, dets):
        """Translucent fills + glowing outlines, drawn with OpenCV (fast, and off the UI thread)."""
        shapes = [(d.poly, self._color(d.cls)) for d in dets if d.poly is not None]
        if not shapes:
            return frame
        fill = frame.copy()
        for poly, col in shapes:
            cv2.fillPoly(fill, [poly], col, cv2.LINE_AA)
        out = cv2.addWeighted(frame, 0.6, fill, 0.4, 0)
        glow = out.copy()
        for poly, col in shapes:
            cv2.polylines(glow, [poly], True, col, 9, cv2.LINE_AA)
        out = cv2.addWeighted(out, 0.6, glow, 0.4, 0)
        for poly, col in shapes:
            cv2.polylines(out, [poly], True, col, 2, cv2.LINE_AA)
        return out

    @staticmethod
    def _draw_poses(frame, poses):
        """Neon skeletons: a soft wide glow pass, then crisp limbs and joints."""
        glow = frame.copy()
        segs = []
        for k in poses:
            for limb, col in SKELETON:
                for i, j in limb:
                    if k[i, 2] >= KP_CONF and k[j, 2] >= KP_CONF:
                        a = (int(k[i, 0]), int(k[i, 1]))
                        b = (int(k[j, 0]), int(k[j, 1]))
                        segs.append((a, b, col))
                        cv2.line(glow, a, b, col, 16, cv2.LINE_AA)
        if not segs:
            return frame
        out = cv2.addWeighted(frame, 0.55, glow, 0.45, 0)
        for a, b, col in segs:
            cv2.line(out, a, b, col, 4, cv2.LINE_AA)
        for k in poses:
            for x, y, c in k:
                if c >= KP_CONF:
                    cv2.circle(out, (int(x), int(y)), 7, (255, 232, 61), 2, cv2.LINE_AA)
                    cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        return out

    @staticmethod
    def _box_distance(depth: np.ndarray, box) -> float | None:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        xa, xb = int(x1 + bw * 0.3), int(x2 - bw * 0.3)
        ya, yb = int(y1 + bh * 0.3), int(y2 - bh * 0.3)
        patch = depth[max(0, ya):max(ya + 1, yb), max(0, xa):max(xa + 1, xb)]
        if patch.size == 0:
            return None
        return float(np.median(patch[::2, ::2]))

    def _depth_overlay(self, frame, depth):
        near, far = self._near_far
        norm = np.clip((depth - near) / max(1e-3, far - near), 0, 1)
        u8 = ((1.0 - norm) * 255).astype(np.uint8)  # near = bright
        color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(frame, 0.30, color, 0.70, 0)

    def _portrait(self, frame, depth, dets):
        """Depth-of-field: keep the nearest subject sharp, melt everything behind it."""
        h, w = frame.shape[:2]
        dists = [d.dist for d in dets if d.dist is not None]
        focus = min(dists) if dists else float(np.percentile(depth[::8, ::8], 10))
        small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        blurred = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4.5), (w, h), interpolation=cv2.INTER_LINEAR)
        a = np.clip((depth - (focus + 0.45)) / 0.7, 0, 1).astype(np.float32)
        a = cv2.GaussianBlur(cv2.resize(a, (w // 4, h // 4)), (0, 0), 2)
        a = cv2.resize(a, (w, h))
        return cv2.blendLinear(blurred, frame, a, 1.0 - a)

    @staticmethod
    def _reset_tracker(model):
        trackers = getattr(getattr(model, "predictor", None), "trackers", None)
        if trackers:
            for t in trackers:
                t.reset()

    def _prepare(self, frame, dets, faces, depth_range, engine, engine_label, cam_fps) -> Packet:
        vw, vh, dpr = self.settings.view
        sh, sw = frame.shape[:2]
        scale = min(vw / sw, vh / sh)
        dw, dh = max(1, int(sw * scale)), max(1, int(sh * scale))
        pw, ph = max(1, int(dw * dpr)), max(1, int(dh * dpr))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_LINEAR if pw >= sw else cv2.INTER_AREA
        img = cv2.resize(rgb, (pw, ph), interpolation=interp)
        return Packet(img, scale, dw, dh, sw, sh, dets, faces, depth_range, engine, engine_label,
                      0.0, cam_fps, dpr)
