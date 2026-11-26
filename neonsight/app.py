"""Main window: wires the sidebar, the video canvas and the inference worker together."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.setswitchinterval(0.001)  # hand the GIL back and forth quickly between UI and inference threads

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget

from .config import ROOT, Config, load_config
from .detector import DetectorWorker, LiveSettings
from .sidebar import Sidebar
from .view import VideoView
from .voice import Voice

SNAP_DIR = ROOT / "snapshots"


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, model, source):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("NeonSight — live object detection")
        self.resize(1500, 900)

        self.settings = LiveSettings(cfg, source)
        self.voice = Voice()
        self.view = VideoView(cfg, self.settings)
        self.sidebar = Sidebar(cfg)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.sidebar)
        self.setCentralWidget(central)

        self.worker = DetectorWorker(cfg, self.settings, model)
        self.worker.packet_ready.connect(self._on_packet)
        self.worker.status.connect(self._on_status)
        self.worker.model_ready.connect(self._on_model_ready)
        self.worker.preset_changed.connect(self._on_preset_changed)

        sb = self.sidebar
        sb.classes_changed.connect(self._on_classes)
        sb.confidence_changed.connect(lambda v: setattr(self.settings, "confidence", v))
        sb.style_changed.connect(self._on_style)
        sb.effect_toggled.connect(self._on_effect)
        sb.prompts_changed.connect(self._on_prompts)
        sb.mission_toggled.connect(self._on_mission)
        sb.mission_skip.connect(self.view.skip_mission)
        sb.model_changed.connect(self._on_model_choice)
        sb.snapshot_requested.connect(self.snapshot)
        sb.camera_requested.connect(self.next_camera)
        self.view.counts_changed.connect(sb.set_counts)
        self.view.fresh_objects.connect(self._announce)
        self.view.mission_event.connect(lambda _k, text: self.voice.say(text))
        self.view.mission_changed.connect(lambda: sb.set_mission(self.view.mission))
        self.view.mission_finished.connect(sb.mission_done)
        self.settings.class_ids = tuple(sb.selected_class_ids())

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(500)
        self._last_ms = 0.0
        self._shortcuts()
        self.worker.start()

    # -------------------------------------------------------------- events
    def _on_packet(self):
        pkt = self.worker.latest()
        if pkt is not None:
            self._last_ms = pkt.infer_ms
            self.view.ingest(pkt)
            if pkt.engine == "yoloe":
                self.sidebar.prompts_live()

    def _on_status(self, text: str):
        self.view.status_text = text
        if self.view.packet is not None:
            self.view.toast(text, 3.5)
        else:
            self.view.update()

    def _on_model_ready(self, text: str):
        self.view.status_text = "Opening camera…"
        self.view.toast(f"⚡ {text}")

    def _on_model_choice(self, preset: str):
        if preset == self.settings.preset:
            return
        self.settings.preset = preset
        self.view.toast(f"Switching to {self.cfg.presets[preset]['label']}…", 1.5)

    def _on_preset_changed(self, preset: str, label: str):
        self.sidebar.set_model(preset, label)
        if preset == self.settings.preset:
            self.view.toast(f"⚡ {preset.capitalize()} model: {label}", 1.8)

    def _on_classes(self, ids: list[int]):
        self.settings.class_ids = tuple(ids)

    def _on_prompts(self, words: list[str]):
        if tuple(words) == self.settings.prompts:
            return
        self.view.register_prompts(words)
        self.settings.prompts = tuple(words)
        if words:
            self.view.toast("✨ Open vocabulary: " + ", ".join(words), 2.8)
        elif self.sidebar.prompt_rows:
            self.view.toast("No known object names to search for", 2.2)
        else:
            self.view.toast("Back to YOLO26 COCO detector", 1.6)

    def _on_style(self, key: str):
        self.view.style = key
        self.view.update()

    def _on_effect(self, key: str, on: bool):
        v = self.view
        if key == "trails":
            v.show_trails = on
        elif key == "spotlight":
            v.spotlight = on
        elif key == "heatmap":
            v.show_heatmap = on
            if on:
                v.heatmap.clear()
        elif key == "privacy":
            self.settings.privacy = on
        elif key == "voice":
            self.voice.enabled = on
            if on:
                self.voice.say("Voice on")
        elif key == "masks":
            self.settings.masks = on
        elif key == "pose":
            self.settings.pose = on
        elif key == "depth":
            self.settings.depth_view = on
        elif key == "portrait":
            self.settings.portrait = on
        elif key == "mirror":
            self.settings.mirror = on
            v.reset_scene()
        v.toast(f"{key.capitalize()} {'on' if on else 'off'}", 1.2)

    def _on_mission(self, on: bool):
        if not on:
            self.view.stop_mission()
        elif not self.view.start_mission():
            self.sidebar.mission_done()
            self.view.toast("Nothing to ask for: tick an object or set mission.objects in config.json", 3)

    def _announce(self, tracks):
        for t in tracks:
            obj = self.view.obj_by_cls.get(t.cls)
            if obj:
                self.voice.say(obj.label, key=obj.coco)

    def _refresh_stats(self):
        v = self.view
        self.sidebar.set_stats(v.display_fps, self._last_ms, len(v.tracks.seen_ids), v.fps_hist)

    # ------------------------------------------------------------- actions
    def snapshot(self):
        SNAP_DIR.mkdir(exist_ok=True)
        path = SNAP_DIR / f"neonsight_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.view.grab().save(str(path))
        self.view.flash()
        self.view.toast(f"📸 Saved snapshots/{path.name}")

    def next_camera(self):
        s = self.settings
        if not isinstance(s.source, int):
            self.view.toast("Camera switching is available in live camera mode", 2.5)
            return
        s.source = (s.source + 1) % 3
        s.source_gen += 1
        self.view.reset_scene()
        self.view.toast(f"🎥 Switching to camera {s.source}…")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _shortcuts(self):
        sb = self.sidebar

        def bind(key, fn):
            QShortcut(QKeySequence(key), self, activated=fn)

        bind("Space", self._toggle_pause)
        bind("S", self.snapshot)
        bind("M", sb.mission_btn.toggle)
        bind("N", self.view.skip_mission)
        bind("C", self.next_camera)
        bind("F", self.toggle_fullscreen)
        bind("Esc", lambda: self.view.close_results() or (self.isFullScreen() and self.showNormal()))
        bind("Tab", lambda: sb.setVisible(not sb.isVisible()))
        bind("Q", self.close)
        for key, effect in (("T", "trails"), ("L", "spotlight"), ("H", "heatmap"), ("P", "privacy"),
                            ("V", "voice")):
            bind(key, sb.effect_buttons[effect].toggle)
        for key, layer in (("G", "masks"), ("K", "pose"), ("D", "depth"), ("B", "portrait")):
            bind(key, sb.layer_buttons[layer].toggle)
        for i, style in enumerate(("neon", "glass", "target"), start=1):
            bind(str(i), lambda s=style: (sb.style_buttons[s].setChecked(True), self._on_style(s)))

    def _toggle_pause(self):
        self.view.paused = not self.view.paused
        self.view.update()

    def closeEvent(self, e):
        self.worker.stop()
        self.voice.shutdown()
        super().closeEvent(e)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="NeonSight — real-time YOLO26 object detection")
    ap.add_argument("--config", default=None, help="path to config JSON (default: config/config.json)")
    ap.add_argument("--source", default=None,
                    help="camera index, or a video/image file (default: camera index from config)")
    return ap.parse_args(argv)


def resolve_source(arg, cfg: Config):
    if arg is None:
        return cfg.camera_index
    if str(arg).isdigit():
        return int(arg)
    if not Path(arg).exists():
        raise SystemExit(f"Source not found: {arg}")
    return str(arg)


def load_model(cfg: Config):
    from ultralytics import YOLO

    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cfg.weights))  # downloads the official weights on first run if missing
    cfg.bind_class_ids(model.names)
    return model


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    source = resolve_source(args.source, cfg)
    t0 = time.perf_counter()
    model = load_model(cfg)
    print(f"[neonsight] {cfg.model_label} ({cfg.preset}) loaded in {time.perf_counter() - t0:.1f}s, "
          f"device={cfg.device}, source={source}")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("NeonSight")
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(7, 9, 16))
    app.setPalette(pal)
    win = MainWindow(cfg, model, source)
    win.show()
    return app.exec()
