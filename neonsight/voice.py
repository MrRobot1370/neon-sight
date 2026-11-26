"""Spoken call-outs through macOS `say`, serialized on a background thread."""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time


class Voice:
    def __init__(self, cooldown_s: float = 7.0):
        self.enabled = False
        self.cooldown_s = cooldown_s
        self._say = shutil.which("say")
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._last: dict[str, float] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str, key: str | None = None, force: bool = False):
        if not self._say or not (self.enabled or force):
            return
        now = time.monotonic()
        if key is not None:
            if now - self._last.get(key, -1e9) < self.cooldown_s:
                return
            self._last[key] = now
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass  # never let speech fall behind the video

    def shutdown(self):
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self):
        while True:
            text = self._q.get()
            if text is None:
                return
            subprocess.run([self._say, "-r", "205", text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
