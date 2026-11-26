"""Eye-candy: activity heatmap, confetti bursts, pulse rings."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen


class Heatmap:
    """Low-res accumulation grid of where objects have been, rendered with the TURBO colormap."""

    def __init__(self, gw: int = 96, gh: int = 54):
        self.gw, self.gh = gw, gh
        self.grid = np.zeros((gh, gw), np.float32)
        k = cv2.getGaussianKernel(11, 2.4)
        self.kernel = (k @ k.T).astype(np.float32)
        self.kernel /= self.kernel.max()
        self._img: QImage | None = None
        self._buf = None

    def clear(self):
        self.grid[:] = 0

    def add(self, nx: float, ny: float, weight: float = 0.12):
        cx, cy = int(nx * self.gw), int(ny * self.gh)
        r = self.kernel.shape[0] // 2
        x0, y0, x1, y1 = cx - r, cy - r, cx + r + 1, cy + r + 1
        gx0, gy0 = max(0, x0), max(0, y0)
        gx1, gy1 = min(self.gw, x1), min(self.gh, y1)
        if gx0 >= gx1 or gy0 >= gy1:
            return
        self.grid[gy0:gy1, gx0:gx1] += weight * self.kernel[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]

    def step(self):
        self.grid *= 0.992
        self._img = None

    def image(self) -> QImage:
        if self._img is None:
            peak = max(1.0, float(self.grid.max()))
            u8 = (np.clip(self.grid / peak, 0, 1) * 255).astype(np.uint8)
            up = cv2.resize(u8, (self.gw * 4, self.gh * 4), interpolation=cv2.INTER_CUBIC)
            color = cv2.applyColorMap(up, cv2.COLORMAP_TURBO)[..., ::-1]
            alpha = np.clip(up.astype(np.float32) * 1.4, 0, 200).astype(np.uint8)
            self._buf = np.ascontiguousarray(np.dstack([color, alpha]))
            h, w = self._buf.shape[:2]
            self._img = QImage(self._buf.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        return self._img


class Particles:
    """Confetti + expanding rings, simulated in widget coordinates."""

    def __init__(self):
        self.parts: list[list] = []  # x, y, vx, vy, life, max_life, size, rot, vrot, QColor
        self.rings: list[list] = []  # x, y, t0, QColor, max_r

    @property
    def active(self) -> bool:
        return bool(self.parts or self.rings)

    def burst(self, x: float, y: float, colors: list[QColor], n: int = 150):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(180, 720)
            life = random.uniform(1.2, 2.2)
            self.parts.append([x, y, math.cos(a) * sp, math.sin(a) * sp - 320, life, life,
                               random.uniform(5, 10), random.uniform(0, 360), random.uniform(-540, 540),
                               random.choice(colors)])

    def ring(self, x: float, y: float, color: QColor, now: float, max_r: float = 180):
        self.rings.append([x, y, now, color, max_r])

    def step(self, dt: float):
        dt = min(dt, 0.05)
        keep = []
        for p in self.parts:
            p[2] *= 0.985
            p[3] = p[3] * 0.985 + 950 * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[7] += p[8] * dt
            p[4] -= dt
            if p[4] > 0:
                keep.append(p)
        self.parts = keep

    def draw(self, painter: QPainter, now: float):
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, _vx, _vy, life, max_life, size, rot, _vr, col in self.parts:
            c = QColor(col)
            c.setAlphaF(min(1.0, life / max_life * 1.6))
            painter.save()
            painter.translate(x, y)
            painter.rotate(rot)
            painter.setBrush(c)
            painter.drawRoundedRect(QRectF(-size / 2, -size / 4, size, size / 2), 1.5, 1.5)
            painter.restore()

        alive = []
        for r in self.rings:
            x, y, t0, col, max_r = r
            t = (now - t0) / 0.7
            if t >= 1:
                continue
            alive.append(r)
            ease = 1 - (1 - t) ** 3
            c = QColor(col)
            c.setAlphaF((1 - t) * 0.9)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(c, 2 + 6 * (1 - t)))
            painter.drawEllipse(QPointF(x, y), max_r * ease, max_r * ease)
        self.rings = alive
