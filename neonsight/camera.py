"""Background frame grabber that always exposes only the newest frame.

Grabbing in its own thread means inference never waits on the camera and
stale frames never pile up in a buffer (the main source of "laggy" webcams).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CameraStream(threading.Thread):
    def __init__(self, source: int | str, width: int, height: int, fps: int):
        super().__init__(daemon=True)
        self.source = source
        self.req_w, self.req_h, self.req_fps = width, height, fps
        self.error: str | None = None
        self.fps = 0.0
        self.size = (0, 0)
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = True
        self.opened = threading.Event()  # set once open succeeded or failed

    # ---------------------------------------------------------------- public
    def wait_latest(self, last_id: int, timeout: float = 0.5):
        """Block until a frame newer than `last_id` exists. Returns (frame, id)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or not self._running, timeout)
            if self._frame_id == last_id:
                return None, last_id
            return self._frame, self._frame_id

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()

    # --------------------------------------------------------------- private
    def _publish(self, frame: np.ndarray):
        with self._cond:
            self._frame = frame
            self._frame_id += 1
            self._cond.notify_all()

    def _tick_fps(self, last_t: float) -> float:
        now = time.perf_counter()
        dt = now - last_t
        if 0 < dt < 1:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1
        return now

    def run(self):
        src = self.source
        if isinstance(src, str) and Path(src).suffix.lower() in IMAGE_EXT:
            self._run_image(src)
        elif isinstance(src, str):
            self._run_video(src)
        else:
            self._run_camera(int(src))

    def _run_camera(self, index: int):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            self.error = (f"Camera {index} unavailable. Allow camera access for your terminal in "
                          "System Settings › Privacy & Security › Camera, then restart.")
            self.opened.set()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        self.opened.set()
        failures, last_t = 0, time.perf_counter()
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 60:
                    self.error = f"Camera {index} stopped delivering frames."
                    break
                time.sleep(0.01)
                continue
            failures = 0
            self.size = (frame.shape[1], frame.shape[0])
            last_t = self._tick_fps(last_t)
            self._publish(frame)
        cap.release()

    def _run_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.error = f"Could not open video: {path}"
            self.opened.set()
            return
        self.opened.set()
        period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        last_t = next_t = time.perf_counter()
        while self._running:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()
            self.size = (frame.shape[1], frame.shape[0])
            last_t = self._tick_fps(last_t)
            self._publish(frame)
        cap.release()

    def _run_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            self.error = f"Could not read image: {path}"
            self.opened.set()
            return
        self.opened.set()
        self.size = (img.shape[1], img.shape[0])
        last_t = time.perf_counter()
        while self._running:
            time.sleep(1 / 30)
            last_t = self._tick_fps(last_t)
            self._publish(img.copy())
