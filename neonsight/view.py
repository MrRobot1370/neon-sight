"""The live video canvas: frame + animated, styled annotations + HUD."""

from __future__ import annotations

import math
import time
from datetime import datetime

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QLinearGradient, QPainter,
                           QPainterPath, QPen, QRadialGradient)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .config import PROMPT_COLORS, Config
from .detector import CUSTOM_BASE, Packet
from .effects import Heatmap, Particles
from .mission import RESULTS_CARD_S, Mission
from .tracks import Track, TrackManager, describe_scene

BG = QColor(7, 9, 16)
INK = QColor(10, 12, 20)
WHITE = QColor(240, 244, 255)
STYLES = ("neon", "glass", "target")



def font(px: int, weight=QFont.Weight.DemiBold, mono: bool = False) -> QFont:
    f = QFont("Menlo" if mono else "Helvetica Neue")
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


class VideoView(QWidget):
    counts_changed = Signal(object)  # {class_key: count}
    mission_changed = Signal()
    fresh_objects = Signal(list)  # list[Track] that just appeared
    mission_event = Signal(str, str)  # kind, text to speak
    mission_finished = Signal()

    def __init__(self, cfg: Config, settings):
        super().__init__()
        self.cfg = cfg
        self.settings = settings
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 400)
        self.setMouseTracking(True)

        self.obj_by_cls = {o.class_id: o for o in cfg.objects}
        self.colors = {o.class_id: QColor(o.color) for o in cfg.objects}
        self.labels = {o.class_id: o.label for o in cfg.objects}
        self.person_cls = next((o.class_id for o in cfg.objects if o.coco == "person"), None)
        self.prompt_keys: list[int] = []
        self.engine: tuple[str, str] | None = None  # (engine, model label) of the last frame

        # toggles (driven by the sidebar)
        self.style = "neon"
        self.show_trails = True
        self.spotlight = False
        self.show_heatmap = False
        self.paused = False

        self.packet: Packet | None = None
        self.qimage: QImage | None = None
        self.tracks = TrackManager()
        self.heatmap = Heatmap()
        self.particles = Particles()
        self.mission = Mission(cfg.mission_rounds, cfg.mission_seconds)
        self._mission_names: dict[int, tuple[str, str]] = {}  # key -> (icon, label), frozen at start
        self._results_rect: QRectF | None = None  # where the results card was last drawn (for clicks)
        self.caption = ""
        self.status_text = "Starting…"
        self.fps_hist: list[float] = []
        self.display_fps = 0.0
        self.paint_ms = 0.0
        self._last_packet_t = 0.0
        self._last_step_t = time.perf_counter()
        self._toast: tuple[str, float] | None = None
        self._flash_t = -10.0
        self._born = time.perf_counter()
        self._n = 0

        self.f_label = font(13, QFont.Weight.Bold)
        self.f_small = font(11, QFont.Weight.DemiBold)
        self.f_mono = font(11, QFont.Weight.Medium, mono=True)
        self.f_hud_big = font(30, QFont.Weight.Black)
        self.f_banner = font(24, QFont.Weight.Black)
        self.f_logo = font(54, QFont.Weight.Black)

        # keeps idle/particle animations running when frames stop
        self._anim = QTimer(self)
        self._anim.timeout.connect(self._maybe_animate)
        self._anim.start(16)

    # ------------------------------------------------------------- inputs
    def register_prompts(self, words: list[str]):
        """Give user-typed (open-vocabulary) objects a label, icon and color."""
        from .config import ObjectSpec

        for k in self.prompt_keys:
            for d in (self.obj_by_cls, self.colors, self.labels):
                d.pop(k, None)
        self.prompt_keys = []
        for i, word in enumerate(words):
            key = CUSTOM_BASE + i
            color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
            self.obj_by_cls[key] = ObjectSpec(coco=word, label=word.title(), icon="✨", color=color, class_id=key)
            self.colors[key] = QColor(color)
            self.labels[key] = word.title()
            self.prompt_keys.append(key)

    def mission_pool(self) -> list[int]:
        """Objects the game asks for: config mission.objects, else the ticked objects + typed words."""
        if self.cfg.mission_class_ids:
            return list(self.cfg.mission_class_ids)
        pool = [c for c in self.settings.class_ids if c != self.person_cls]
        if self.settings.prompts:
            pool += self.prompt_keys
        return pool or [o.class_id for o in self.cfg.objects if o.class_id != self.person_cls]

    def ingest(self, pkt: Packet):
        if self.paused:
            return
        now = time.perf_counter()
        if self._last_packet_t:
            inst = 1.0 / max(1e-3, now - self._last_packet_t)
            self.display_fps = inst if not self.display_fps else self.display_fps * 0.9 + inst * 0.1
        self._last_packet_t = now
        if (pkt.engine, pkt.engine_label) != self.engine:  # new model: old track IDs mean nothing
            self.engine = (pkt.engine, pkt.engine_label)
            self.tracks.reset()
        self.packet = pkt
        self.qimage = QImage(pkt.image.data, pkt.image.shape[1], pkt.image.shape[0],
                             pkt.image.shape[1] * 3, QImage.Format.Format_RGB888)
        self.qimage.setDevicePixelRatio(pkt.dpr)

        fresh = self.tracks.update(pkt.detections, now)
        alive = self.tracks.alive_tracks()
        self.heatmap.step()
        for t in alive:
            cx, cy = t.center
            self.heatmap.add(cx / pkt.src_w, cy / pkt.src_h)

        self._n += 1
        if self._n % 3 == 0:
            self.counts_changed.emit(dict(self.tracks.counts()))
            self.caption = describe_scene(alive, self.labels, self.person_cls)
        if self._n % 10 == 0:
            self.fps_hist = (self.fps_hist + [self.display_fps])[-60:]
        if fresh:
            self.fresh_objects.emit(fresh)

        ev = self.mission.update(alive, set(self.obj_by_cls), now)
        if ev:
            self._on_mission_event(ev, now)
        self.update()

    def _on_mission_event(self, ev, now):
        kind, track = ev
        if kind == "found" and track is not None:
            r = self._map_box(track.box)
            col = self.colors.get(track.cls, WHITE)
            c = r.center()
            self.particles.burst(c.x(), c.y(), [col, QColor("#FFFFFF"), QColor("#FFD23D"), QColor("#3DE8FF")])
            self.particles.ring(c.x(), c.y(), col, now, max(r.width(), r.height()) * 0.9)
            self.mission_event.emit("found", f"Got it! {self._mission_label(track.cls)} in "
                                             f"{self.mission.last.secs:.1f} seconds")
        elif kind == "timeout":
            self.mission_event.emit("timeout", "Time's up!")
        elif kind == "next" and self.mission.active:
            self.mission_event.emit("next", f"Show me a {self._mission_label(self.mission.target)}")
        elif kind == "done":
            m = self.mission
            self.settings.mission_ids = ()
            self.mission_event.emit("done", f"Mission complete! You found {m.found} of {m.total}. "
                                            f"{m.score} points.")
            self.mission_finished.emit()
        self.mission_changed.emit()

    def _mission_label(self, key) -> str:
        return self._mission_names.get(key, ("", "?"))[1]

    def start_mission(self) -> bool:
        pool = self.mission_pool()
        self._mission_names = {k: (o.icon, o.label) for k, o in self.obj_by_cls.items() if k in pool}
        pool = [k for k in pool if k in self._mission_names]
        if not self.mission.start(pool, time.perf_counter()):
            return False
        self.settings.mission_ids = tuple(k for k in pool if k < CUSTOM_BASE)
        m = self.mission
        self.mission_event.emit("next", f"Mission: {m.total} rounds. Show me a {self._mission_label(m.target)}")
        self.mission_changed.emit()
        return True

    def stop_mission(self):
        self.mission.stop()
        self.settings.mission_ids = ()
        self.mission_changed.emit()

    def close_results(self) -> bool:
        """Close the mission results card. Returns True if it was open."""
        if self._results_rect is None or not self.mission.showing_results(time.perf_counter()):
            return False
        self.mission.dismiss_results()
        self._results_rect = None
        self.unsetCursor()
        self.update()
        return True

    def mousePressEvent(self, e):
        if self._results_rect is not None and self._results_rect.contains(e.position()):
            self.close_results()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        over = self._results_rect is not None and self._results_rect.contains(e.position())
        if over:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(e)

    def skip_mission(self):
        ev = self.mission.skip(time.perf_counter())
        if ev:
            self.mission_changed.emit()

    def toast(self, text: str, secs: float = 2.2):
        self._toast = (text, time.perf_counter() + secs)
        self.update()

    def flash(self):
        self._flash_t = time.perf_counter()

    def reset_scene(self):
        self.tracks.reset()
        self.heatmap.clear()

    # ---------------------------------------------------------- geometry
    def _video_rect(self) -> QRectF:
        p = self.packet
        if p is None:
            return QRectF(self.rect())
        return QRectF((self.width() - p.disp_w) / 2, (self.height() - p.disp_h) / 2, p.disp_w, p.disp_h)

    def _map_box(self, box) -> QRectF:
        p = self.packet
        vr = self._video_rect()
        s = p.scale
        return QRectF(vr.x() + box[0] * s, vr.y() + box[1] * s, (box[2] - box[0]) * s, (box[3] - box[1]) * s)

    def _map_pt(self, x, y) -> QPointF:
        vr = self._video_rect()
        s = self.packet.scale
        return QPointF(vr.x() + x * s, vr.y() + y * s)

    def resizeEvent(self, e):
        self._sync_view_size()
        super().resizeEvent(e)

    def _sync_view_size(self):
        view = (max(1, self.width()), max(1, self.height()), float(self.devicePixelRatioF()))
        if self.settings.view != view:
            self.settings.view = view

    def _maybe_animate(self):
        now = time.perf_counter()
        dt = now - self._last_step_t
        self._last_step_t = now
        if self.particles.active:
            self.particles.step(dt)
            self.update()
        elif now - self._last_packet_t > 0.1 or self.paused:
            self.update()  # idle screen / paused overlay still animates

    # ------------------------------------------------------------ painting
    def paintEvent(self, _e):
        t0 = time.perf_counter()
        self._sync_view_size()
        now = t0
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                         | QPainter.RenderHint.SmoothPixmapTransform)
        self._paint_background(p)

        if self.packet is None or self.qimage is None:
            self._paint_idle(p, now)
            self._paint_toast(p, now)
            p.end()
            return

        vr = self._video_rect()
        p.drawImage(vr.topLeft(), self.qimage)
        p.setClipRect(vr)

        if self.show_heatmap:
            p.setOpacity(0.85)
            p.drawImage(vr, self.heatmap.image())
            p.setOpacity(1.0)

        visible = [(t, a, age, self._map_box(t.box)) for t, a, age in self.tracks.visible(now)]
        if self.spotlight and visible:
            self._paint_spotlight(p, vr, visible)
        if self.show_trails:
            for t, a, _age, _r in visible:
                self._paint_trail(p, t, a)


        target = self.mission.target if self.mission.active and self.mission.state == "hunt" else None
        for t, a, age, r in sorted(visible, key=lambda v: -v[3].width() * v[3].height()):
            r, a = self._pop(r, a, age)
            col = self.colors.get(t.cls, WHITE)
            p.setBrush(Qt.BrushStyle.NoBrush)  # never inherit a fill from the previous draw
            if target is not None and t.cls == target:
                self._paint_target_lock(p, r, col, now)
            {"neon": self._box_neon, "glass": self._box_glass, "target": self._box_target}[self.style](
                p, r, col, a, t, now)

        self._paint_faces(p)
        self._paint_scanline(p, vr, now)
        p.setClipping(False)

        self._paint_frame_corners(p, vr)
        self._paint_hud(p, vr, now)
        self._paint_caption(p, vr)
        if self.mission.active:
            self._paint_mission(p, vr, now)
        elif self.mission.showing_results(now):
            self._paint_mission_results(p, vr, now)
        elif self._results_rect is not None:
            self._results_rect = None
            self.unsetCursor()
        self.particles.draw(p, now)
        self._paint_flash(p, now)
        self._paint_toast(p, now)
        if self.paused:
            self._paint_paused(p, vr)
        p.end()
        self.paint_ms = self.paint_ms * 0.9 + (time.perf_counter() - t0) * 1000 * 0.1

    # --- background / idle
    def _paint_background(self, p: QPainter):
        g = QRadialGradient(QPointF(self.width() * 0.5, self.height() * 0.45), max(self.width(), self.height()) * 0.7)
        g.setColorAt(0, QColor(18, 24, 44))
        g.setColorAt(1, BG)
        p.fillRect(self.rect(), g)

    def _paint_idle(self, p: QPainter, now: float):
        cx, cy = self.width() / 2, self.height() / 2
        t = now - self._born
        # spinning dual arcs
        for i, (col, rad, speed) in enumerate(((QColor("#3DE8FF"), 70, 220), (QColor("#FF3D9A"), 56, -300))):
            p.setPen(QPen(col, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(QRectF(cx - rad, cy - 130 - rad, rad * 2, rad * 2), int(t * speed * 16) % 5760, 100 * 16)
        p.setFont(self.f_logo)
        grad = QLinearGradient(cx - 200, 0, cx + 200, 0)
        grad.setColorAt(0, QColor("#3DE8FF"))
        grad.setColorAt(1, QColor("#FF3D9A"))
        p.setPen(QPen(grad, 1))
        p.drawText(QRectF(0, cy - 20, self.width(), 70), Qt.AlignmentFlag.AlignCenter, "NEONSIGHT")
        p.setFont(font(15, QFont.Weight.Medium))
        p.setPen(with_alpha(WHITE, 0.75))
        p.drawText(QRectF(40, cy + 60, self.width() - 80, 60),
                   Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, self.status_text)

    # --- per-object
    @staticmethod
    def _pop(r: QRectF, a: float, age: float):
        """Pop-in: boxes snap in from slightly larger and fade up."""
        if age >= 0.32:
            return r, a
        k = min(1.0, age / 0.32)
        grow = 0.28 * (1 - k) ** 3
        dx, dy = r.width() * grow / 2, r.height() * grow / 2
        return r.adjusted(-dx, -dy, dx, dy), a * (0.25 + 0.75 * k)

    def _paint_spotlight(self, p: QPainter, vr: QRectF, visible):
        holes = QPainterPath()
        holes.setFillRule(Qt.FillRule.WindingFill)
        for _t, a, _age, r in visible:
            if a > 0.2:
                holes.addRoundedRect(r.adjusted(-14, -14, 14, 14), 18, 18)
        dim = QPainterPath()
        dim.addRect(vr)
        p.fillPath(dim.subtracted(holes.simplified()), QColor(4, 6, 12, 185))

    def _paint_trail(self, p: QPainter, t: Track, a: float):
        pts = list(t.trail)
        if len(pts) < 3:
            return
        col = self.colors.get(t.cls, WHITE)
        n = len(pts)
        prev = self._map_pt(*pts[0])
        for i in range(1, n):
            cur = self._map_pt(*pts[i])
            k = i / n
            p.setPen(QPen(with_alpha(col, 0.85 * k * a), 1 + 4.5 * k, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(prev, cur)
            prev = cur
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(col, a))
        p.drawEllipse(prev, 3.5, 3.5)

    def _label_parts(self, t: Track):
        obj = self.obj_by_cls.get(t.cls)
        name = f"{obj.icon}  {obj.label}" if obj else str(t.cls)
        conf = f"{int(round(t.conf * 100))}%"
        if t.dist is not None:
            conf += f" · {t.dist:.1f}m"
        return name, conf

    def _pill(self, p: QPainter, x: float, y: float, col: QColor, a: float, t: Track, bounds: QRectF):
        """Colored label pill with a dark confidence segment. (x, y) = bottom-left anchor."""
        name, conf = self._label_parts(t)
        fm = QFontMetricsF(self.f_label)
        fs = QFontMetricsF(self.f_mono)
        h = 24.0
        wn = fm.horizontalAdvance(name) + 20
        wc = fs.horizontalAdvance(conf) + 14
        w = wn + wc
        x = min(max(x, bounds.left() + 4), bounds.right() - w - 4)
        top = y - h
        if top < bounds.top() + 4:
            top = bounds.top() + 4
        r = QRectF(x, top, w, h)
        path = QPainterPath()
        path.addRoundedRect(r, h / 2, h / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.fillPath(path, with_alpha(col, 0.92 * a))
        seg = QPainterPath()
        seg.addRoundedRect(QRectF(x + wn - 2, top + 3, wc - 1, h - 6), (h - 6) / 2, (h - 6) / 2)
        p.fillPath(seg, with_alpha(INK, 0.55 * a))
        p.setFont(self.f_label)
        p.setPen(with_alpha(INK, a))
        p.drawText(QRectF(x + 10, top, wn - 10, h), Qt.AlignmentFlag.AlignVCenter, name)
        p.setFont(self.f_mono)
        p.setPen(with_alpha(WHITE, a))
        p.drawText(QRectF(x + wn - 2, top, wc - 1, h), Qt.AlignmentFlag.AlignCenter, conf)
        return r

    def _box_neon(self, p: QPainter, r: QRectF, col: QColor, a: float, t: Track, now: float):
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, with_alpha(col, 0.22 * a))
        g.setColorAt(1, with_alpha(col, 0.03 * a))
        body = QPainterPath()
        body.addRoundedRect(r, 8, 8)
        p.fillPath(body, g)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(col, 0.45 * a), 1.2))
        p.drawPath(body)

        L = max(12.0, min(44.0, min(r.width(), r.height()) * 0.24))
        corners = QPainterPath()
        x0, y0, x1, y1 = r.left(), r.top(), r.right(), r.bottom()
        for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            corners.moveTo(cx, cy + dy * L)
            corners.lineTo(cx, cy)
            corners.lineTo(cx + dx * L, cy)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(col, 0.28 * a), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(corners)
        p.setPen(QPen(with_alpha(col, a), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(corners)
        self._pill(p, r.left(), r.top() - 6, col, a, t, self._video_rect())
        if t.key is not None and r.height() > 50:
            p.setFont(self.f_mono)
            p.setPen(with_alpha(col, 0.85 * a))
            p.drawText(QRectF(r.right() - 70, r.bottom() - 20, 62, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"ID {t.key}")

    def _box_glass(self, p: QPainter, r: QRectF, col: QColor, a: float, t: Track, now: float):
        body = QPainterPath()
        body.addRoundedRect(r, 16, 16)
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, QColor(255, 255, 255, int(46 * a)))
        g.setColorAt(0.5, QColor(255, 255, 255, int(10 * a)))
        g.setColorAt(1, with_alpha(col, 0.16 * a))
        p.fillPath(body, g)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(col, 0.35 * a), 6))
        p.drawPath(body)
        p.setPen(QPen(with_alpha(col, 0.95 * a), 2))
        p.drawPath(body)

        name, conf = self._label_parts(t)
        fm = QFontMetricsF(self.f_label)
        w = fm.horizontalAdvance(name) + 34
        tag = QRectF(r.left() + 10, r.top() + 10, min(w, max(40.0, r.width() - 20)), 26)
        if r.height() < 60:
            tag.moveBottom(r.top() - 6)
        tp = QPainterPath()
        tp.addRoundedRect(tag, 13, 13)
        p.fillPath(tp, QColor(8, 10, 18, int(185 * a)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(col, a))
        p.drawEllipse(QPointF(tag.left() + 13, tag.center().y()), 4.5, 4.5)
        p.setFont(self.f_label)
        p.setPen(with_alpha(WHITE, a))
        p.drawText(tag.adjusted(24, 0, -6, 0), Qt.AlignmentFlag.AlignVCenter, name)

        # confidence ring, top-right
        rad = 15.0
        c = QPointF(r.right() - rad - 10, r.top() + rad + 10) if r.height() >= 60 else QPointF(r.right() - rad, r.top() - rad - 6)
        p.setBrush(QColor(8, 10, 18, int(185 * a)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(c, rad + 3, rad + 3)
        ring = QRectF(c.x() - rad, c.y() - rad, rad * 2, rad * 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, int(40 * a)), 3))
        p.drawEllipse(ring)
        p.setPen(QPen(with_alpha(col, a), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, -int(360 * 16 * t.conf))
        p.setFont(font(10, QFont.Weight.Bold))
        p.setPen(with_alpha(WHITE, a))
        p.drawText(ring, Qt.AlignmentFlag.AlignCenter, conf.rstrip("%"))

    def _box_target(self, p: QPainter, r: QRectF, col: QColor, a: float, t: Track, now: float):
        c = r.center()
        R = max(18.0, min(150.0, min(r.width(), r.height()) * 0.42))
        # faint box
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(col, 0.25 * a), 1, Qt.PenStyle.DashLine))
        p.drawRect(r)
        # rotating segmented ring
        p.save()
        p.translate(c)
        p.rotate((now * 70 + (t.key or 0) * 37) % 360)
        p.setPen(QPen(with_alpha(col, 0.3 * a), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        ring = QRectF(-R, -R, 2 * R, 2 * R)
        for k in range(3):
            p.drawArc(ring, (k * 120) * 16, 80 * 16)
        p.setPen(QPen(with_alpha(col, a), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for k in range(3):
            p.drawArc(ring, (k * 120) * 16, 80 * 16)
        p.restore()
        # counter-rotating inner ring + ticks
        p.save()
        p.translate(c)
        p.rotate(-(now * 110) % 360)
        r2 = R * 0.62
        p.setPen(QPen(with_alpha(col, 0.7 * a), 1.5, Qt.PenStyle.DotLine))
        p.drawEllipse(QPointF(0, 0), r2, r2)
        p.restore()
        p.setPen(QPen(with_alpha(col, a), 2))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(c.x() + dx * (R + 4), c.y() + dy * (R + 4)),
                       QPointF(c.x() + dx * (R + 14), c.y() + dy * (R + 14)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(col, a))
        p.drawEllipse(c, 3, 3)

        # callout leader line + label
        name, conf = self._label_parts(t)
        text = f"{name}   {conf}"
        fm = QFontMetricsF(self.f_label)
        tw = fm.horizontalAdvance(text) + 12
        vr = self._video_rect()
        right = c.x() + R + 30 + tw < vr.right()
        sx = 1 if right else -1
        p1 = QPointF(c.x() + sx * R * 0.707, c.y() - R * 0.707)
        p2 = QPointF(p1.x() + sx * 26, max(vr.top() + 30, p1.y() - 26))
        p3 = QPointF(p2.x() + sx * tw, p2.y())
        p.setPen(QPen(with_alpha(col, a), 1.6))
        p.drawLine(p1, p2)
        p.drawLine(p2, p3)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(with_alpha(col, a))
        p.drawEllipse(p1, 3, 3)
        box = QRectF(min(p2.x(), p3.x()), p2.y() - 26, tw, 24)
        bg = QPainterPath()
        bg.addRoundedRect(box, 6, 6)
        p.fillPath(bg, QColor(6, 8, 14, int(170 * a)))
        p.setFont(self.f_label)
        p.setPen(with_alpha(col, a))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_target_lock(self, p: QPainter, r: QRectF, col: QColor, now: float):
        """Pulsing halo on objects that match the current mission target."""
        pulse = 0.5 + 0.5 * math.sin(now * 9)
        grow = 6 + 8 * pulse
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(QColor("#FFFFFF"), 0.35 + 0.5 * pulse), 2.5, Qt.PenStyle.DashLine))
        p.drawRoundedRect(r.adjusted(-grow, -grow, grow, grow), 14, 14)

    def _paint_faces(self, p: QPainter):
        if not self.packet.faces:
            return
        p.setFont(font(10, QFont.Weight.Bold))
        for x0, y0, x1, y1 in self.packet.faces:
            tl, br = self._map_pt(x0, y0), self._map_pt(x1, y1)
            r = QRectF(tl, br)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 140), 1.5, Qt.PenStyle.DashLine))
            p.drawRoundedRect(r, 10, 10)
            tag = QRectF(r.center().x() - 40, r.bottom() + 4, 80, 18)
            bg = QPainterPath()
            bg.addRoundedRect(tag, 9, 9)
            p.fillPath(bg, QColor(8, 10, 18, 200))
            p.setPen(WHITE)
            p.drawText(tag, Qt.AlignmentFlag.AlignCenter, "🔒 PRIVATE")

    # --- overlays
    def _paint_scanline(self, p: QPainter, vr: QRectF, now: float):
        period, dur = 7.0, 1.6
        t = ((now - self._born) % period) / dur
        if t > 1:
            return
        y = vr.top() + vr.height() * (t * t * (3 - 2 * t))
        g = QLinearGradient(0, y - 90, 0, y)
        g.setColorAt(0, QColor(61, 232, 255, 0))
        g.setColorAt(1, QColor(61, 232, 255, 38))
        p.fillRect(QRectF(vr.left(), y - 90, vr.width(), 90), g)
        p.setPen(QPen(QColor(61, 232, 255, 150), 1.4))
        p.drawLine(QPointF(vr.left(), y), QPointF(vr.right(), y))

    def _paint_frame_corners(self, p: QPainter, vr: QRectF):
        L, m = 26, 12
        p.setPen(QPen(QColor(255, 255, 255, 110), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        x0, y0, x1, y1 = vr.left() + m, vr.top() + m, vr.right() - m, vr.bottom() - m
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * L, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * L))

    def _glass_panel(self, p: QPainter, r: QRectF, radius: float = 14, alpha: int = 165):
        path = QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p.fillPath(path, QColor(8, 11, 20, alpha))
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _paint_hud(self, p: QPainter, vr: QRectF, now: float):
        pk = self.packet
        panel = QRectF(vr.left() + 26, vr.top() + 26, 236, 92)
        self._glass_panel(p, panel)
        # live dot
        blink = 0.55 + 0.45 * math.sin(now * 5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 61, 90, int(255 * blink)))
        p.drawEllipse(QPointF(panel.left() + 18, panel.top() + 20), 5, 5)
        p.setFont(self.f_small)
        p.setPen(WHITE)
        p.drawText(QRectF(panel.left() + 30, panel.top() + 10, 200, 20), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetricsF(self.f_small).elidedText(f"LIVE  ·  {pk.engine_label}", Qt.TextElideMode.ElideRight, 200))
        p.setFont(self.f_hud_big)
        fps_col = QColor("#3DFF8A") if self.display_fps >= 29 else QColor("#FFC23D")
        p.setPen(fps_col)
        p.drawText(QRectF(panel.left() + 14, panel.top() + 30, 120, 40), Qt.AlignmentFlag.AlignVCenter,
                   f"{self.display_fps:4.0f}")
        p.setFont(self.f_small)
        p.setPen(with_alpha(WHITE, 0.6))
        p.drawText(QRectF(panel.left() + 82, panel.top() + 40, 40, 24), Qt.AlignmentFlag.AlignVCenter, "FPS")
        capacity = 1000.0 / pk.infer_ms if pk.infer_ms > 0 else 0
        p.setFont(self.f_mono)
        p.setPen(with_alpha(WHITE, 0.85))
        p.drawText(QRectF(panel.left() + 124, panel.top() + 34, 108, 18), Qt.AlignmentFlag.AlignVCenter,
                   f"AI {pk.infer_ms:5.1f} ms")
        p.drawText(QRectF(panel.left() + 124, panel.top() + 52, 108, 18), Qt.AlignmentFlag.AlignVCenter,
                   f"cap {capacity:4.0f} fps")
        p.setPen(with_alpha(WHITE, 0.55))
        p.drawText(QRectF(panel.left() + 14, panel.top() + 70, 220, 16), Qt.AlignmentFlag.AlignVCenter,
                   f"{pk.src_w}×{pk.src_h}  ·  {len(self.tracks.alive_tracks())} objects  ·  {len(self.tracks.seen_ids)} seen")

        if pk.depth_range:
            self._paint_depth_legend(p, vr, pk.depth_range)

        # clock, top-right
        clock = QRectF(vr.right() - 26 - 112, vr.top() + 26, 112, 34)
        self._glass_panel(p, clock, 12)
        p.setFont(font(15, QFont.Weight.Bold, mono=True))
        p.setPen(WHITE)
        p.drawText(clock, Qt.AlignmentFlag.AlignCenter, datetime.now().strftime("%H:%M:%S"))

    def _paint_depth_legend(self, p: QPainter, vr: QRectF, rng):
        near, far = rng
        r = QRectF(vr.left() + 26, vr.top() + 128, 236, 46)
        self._glass_panel(p, r, 12)
        bar = QRectF(r.left() + 14, r.top() + 26, r.width() - 28, 8)
        g = QLinearGradient(bar.topLeft(), bar.topRight())
        for stop, hexcol in ((0, "#FCFFA4"), (0.25, "#F98C0A"), (0.5, "#BC3754"), (0.75, "#57106E"), (1, "#000004")):
            g.setColorAt(stop, QColor(hexcol))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(bar, 4, 4)
        p.setFont(self.f_mono)
        p.setPen(with_alpha(WHITE, 0.85))
        p.drawText(QRectF(r.left() + 14, r.top() + 6, 100, 16), Qt.AlignmentFlag.AlignVCenter, f"{near:.1f} m")
        p.drawText(QRectF(r.right() - 114, r.top() + 6, 100, 16),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{far:.1f} m")
        p.setFont(self.f_small)
        p.drawText(QRectF(r.left(), r.top() + 6, r.width(), 16), Qt.AlignmentFlag.AlignCenter, "DEPTH")

    def _paint_caption(self, p: QPainter, vr: QRectF):
        if not self.caption:
            return
        text = f"🧠  {self.caption}"
        f = font(15, QFont.Weight.DemiBold)
        fm = QFontMetricsF(f)
        w = min(vr.width() - 60, fm.horizontalAdvance(text) + 40)
        r = QRectF(vr.center().x() - w / 2, vr.bottom() - 26 - 40, w, 40)
        self._glass_panel(p, r, 20)
        p.setFont(f)
        p.setPen(WHITE)
        p.drawText(r.adjusted(20, 0, -20, 0), Qt.AlignmentFlag.AlignCenter,
                   fm.elidedText(text, Qt.TextElideMode.ElideRight, w - 40))

    MISSION_COLORS = {"found": QColor("#3DFF8A"), "timeout": QColor("#FF4D5E"), "skipped": QColor("#8a95b3")}

    def _paint_mission(self, p: QPainter, vr: QRectF, now: float):
        m = self.mission
        key = m.target
        icon, label = self._mission_names.get(key, ("", "?"))
        col = self.colors.get(key, WHITE)
        last = m.last if m.state != "hunt" else None
        w, h = min(600.0, vr.width() - 340), 96.0
        r = QRectF(vr.center().x() - w / 2, vr.top() + 26, w, h)
        if m.state == "found":
            accent, title, sub = self.MISSION_COLORS["found"], f"FOUND IT!  {icon} {label}", \
                f"{last.secs:.1f}s  +{last.points}"
        elif m.state == "timeout":
            accent, title, sub = self.MISSION_COLORS["timeout"], "TIME'S UP", f"no {label.lower()}"
        elif m.state == "skipped":
            accent, title, sub = self.MISSION_COLORS["skipped"], f"SKIPPED  {icon} {label}", ""
        else:
            accent, title = col, f"Show me a  {icon} {label.upper()}"
            sub = f"{m.time_left(now):4.1f}s left"
        self._glass_panel(p, r, 18, 200)
        glow = QPainterPath()
        glow.addRoundedRect(r, 18, 18)
        p.setPen(QPen(with_alpha(accent, 0.7 + 0.3 * math.sin(now * 6)), 2))
        p.drawPath(glow)

        # top line: round counter, one dot per round, score
        p.setFont(self.f_small)
        p.setPen(with_alpha(accent, 1))
        head = f"ROUND {m.round} / {m.total}"
        p.drawText(QRectF(r.left() + 20, r.top() + 8, 120, 18), Qt.AlignmentFlag.AlignVCenter, head)
        p.setPen(with_alpha(WHITE, 0.8))
        p.drawText(QRectF(r.right() - 220, r.top() + 8, 200, 18),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"{m.found} found  ·  SCORE {m.score}")
        x0 = r.left() + 30 + QFontMetricsF(self.f_small).horizontalAdvance(head)
        room = r.right() - 240 - x0
        step = max(6.0, min(16.0, room / max(1, m.total)))
        for i in range(m.total):
            c = QPointF(x0 + step * i + step / 2, r.top() + 17)
            rad = min(5.0, step * 0.35)
            if i < len(m.results):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self.MISSION_COLORS[m.results[i].outcome])
                p.drawEllipse(c, rad, rad)
            elif i == m.index:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(with_alpha(col, 0.6 + 0.4 * math.sin(now * 8)), 2))
                p.drawEllipse(c, rad + 1, rad + 1)
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(with_alpha(WHITE, 0.25), 1.2))
                p.drawEllipse(c, rad, rad)
        p.setBrush(Qt.BrushStyle.NoBrush)

        p.setFont(self.f_banner)
        p.setPen(WHITE)
        p.drawText(QRectF(r.left() + 20, r.top() + 30, w - 150, 34), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetricsF(self.f_banner).elidedText(title, Qt.TextElideMode.ElideRight, w - 150))
        p.setFont(self.f_mono)
        p.setPen(with_alpha(WHITE, 0.75))
        p.drawText(QRectF(r.left() + 20, r.top() + 30, w - 40, 34),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, sub)
        if m.state == "hunt":
            frac = m.time_left(now) / m.seconds
            bar = QRectF(r.left() + 20, r.bottom() - 18, w - 130, 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawRoundedRect(bar, 2.5, 2.5)
            p.setBrush(accent if frac > 0.25 else QColor("#FF4D5E"))
            p.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * frac, bar.height()), 2.5, 2.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setFont(self.f_small)
            p.setPen(with_alpha(WHITE, 0.55))
            p.drawText(QRectF(r.right() - 100, r.bottom() - 26, 80, 20),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, "N = skip")

    def _paint_mission_results(self, p: QPainter, vr: QRectF, now: float):
        m = self.mission
        age = now - m.finished_at
        a = min(1.0, age / 0.3, (RESULTS_CARD_S - age) / 0.8)
        if a <= 0:
            return
        row_h = max(18.0, min(30.0, (vr.height() - 220) / max(1, m.total)))
        w = min(480.0, vr.width() - 60)
        h = 128 + row_h * m.total + 36
        r = QRectF(vr.center().x() - w / 2, vr.center().y() - h / 2, w, h)
        self._results_rect = r
        p.setOpacity(a)
        self._glass_panel(p, r, 20, 225)
        edge = QPainterPath()
        edge.addRoundedRect(r, 20, 20)
        g = QLinearGradient(r.topLeft(), r.topRight())
        g.setColorAt(0, QColor("#3DE8FF"))
        g.setColorAt(1, QColor("#FF3D9A"))
        p.setPen(QPen(QBrush(g), 2))
        p.drawPath(edge)

        p.setFont(self.f_banner)
        p.setPen(WHITE)
        p.drawText(QRectF(r.left(), r.top() + 18, w, 36), Qt.AlignmentFlag.AlignCenter, "MISSION COMPLETE")
        p.setFont(self.f_label)
        p.setPen(QColor("#7CFFCB"))
        best = f"   ·   best {m.best:.1f}s" if m.best is not None else ""
        p.drawText(QRectF(r.left(), r.top() + 58, w, 24), Qt.AlignmentFlag.AlignCenter,
                   f"{m.found} / {m.total} found   ·   {m.score} points{best}")
        p.setPen(QPen(with_alpha(WHITE, 0.12), 1))
        p.drawLine(QPointF(r.left() + 24, r.top() + 98), QPointF(r.right() - 24, r.top() + 98))

        y = r.top() + 108
        for i, res in enumerate(m.results):
            icon, label = self._mission_names.get(res.target, ("", "?"))
            row = QRectF(r.left() + 28, y, w - 56, row_h)
            p.setFont(self.f_small)
            p.setPen(with_alpha(WHITE, 0.45))
            p.drawText(QRectF(row.left(), row.top(), 26, row_h), Qt.AlignmentFlag.AlignVCenter, f"{i + 1}")
            p.setFont(self.f_label)
            p.setPen(WHITE)
            p.drawText(QRectF(row.left() + 26, row.top(), 200, row_h), Qt.AlignmentFlag.AlignVCenter,
                       f"{icon}  {label}")
            outcome = {"found": f"✓  {res.secs:.1f} s" if res.secs is not None else "✓",
                       "timeout": "✗  time's up", "skipped": "⏭  skipped"}[res.outcome]
            p.setPen(self.MISSION_COLORS[res.outcome])
            p.drawText(QRectF(row.left() + 200, row.top(), 140, row_h), Qt.AlignmentFlag.AlignVCenter, outcome)
            p.setPen(WHITE if res.points else with_alpha(WHITE, 0.35))
            p.drawText(row, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"+{res.points}" if res.points else "0")
            y += row_h
        p.setFont(self.f_small)
        p.setPen(with_alpha(WHITE, 0.5))
        p.drawText(QRectF(r.left(), r.bottom() - 34, w, 24), Qt.AlignmentFlag.AlignCenter,
                   "Click or press Esc to close  ·  M to play again")
        # close button
        c = QPointF(r.right() - 26, r.top() + 26)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 28))
        p.drawEllipse(c, 13, 13)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(with_alpha(WHITE, 0.85), 2))
        p.drawLine(QPointF(c.x() - 5, c.y() - 5), QPointF(c.x() + 5, c.y() + 5))
        p.drawLine(QPointF(c.x() - 5, c.y() + 5), QPointF(c.x() + 5, c.y() - 5))
        p.setOpacity(1)

    def _paint_flash(self, p: QPainter, now: float):
        t = (now - self._flash_t) / 0.35
        if 0 <= t < 1:
            p.fillRect(self.rect(), QColor(255, 255, 255, int(200 * (1 - t) ** 2)))

    def _paint_toast(self, p: QPainter, now: float):
        if not self._toast:
            return
        text, until = self._toast
        left = until - now
        if left <= 0:
            self._toast = None
            return
        a = min(1.0, left / 0.4)
        f = font(14, QFont.Weight.Bold)
        w = QFontMetricsF(f).horizontalAdvance(text) + 44
        r = QRectF(self.width() / 2 - w / 2, self.height() - 140, w, 38)
        p.setOpacity(a)
        self._glass_panel(p, r, 19, 220)
        p.setFont(f)
        p.setPen(WHITE)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        p.setOpacity(1)

    def _paint_paused(self, p: QPainter, vr: QRectF):
        p.fillRect(vr, QColor(0, 0, 0, 90))
        p.setFont(self.f_banner)
        p.setPen(WHITE)
        p.drawText(vr, Qt.AlignmentFlag.AlignCenter, "⏸  PAUSED  —  press Space")
