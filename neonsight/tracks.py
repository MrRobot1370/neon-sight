"""Per-object visual state: smoothed boxes, motion trails, enter/exit animation."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .detector import Detection

SMOOTHING = 0.6  # 1.0 = raw boxes, lower = smoother but more lag
LINGER_S = 0.35  # how long a lost object fades out
TRAIL_LEN = 26


class Track:
    __slots__ = ("key", "cls", "conf", "box", "first_seen", "last_seen", "trail", "tracked", "alive", "dist")

    def __init__(self, key, det: Detection, now: float, tracked: bool):
        self.key = key
        self.cls = det.cls
        self.conf = det.conf
        self.box = np.array(det.box, dtype=np.float32)
        self.first_seen = now if tracked else now - 10.0  # untracked boxes skip the pop-in
        self.last_seen = now
        self.trail: deque = deque(maxlen=TRAIL_LEN)
        self.tracked = tracked
        self.alive = True
        self.dist = det.dist

    @property
    def center(self):
        b = self.box
        return float((b[0] + b[2]) * 0.5), float((b[1] + b[3]) * 0.5)

    def area(self):
        b = self.box
        return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


class TrackManager:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.transient: list[Track] = []
        self.seen_ids: set[int] = set()

    def reset(self):
        self.tracks.clear()
        self.transient = []

    def update(self, dets: list[Detection], now: float) -> list[Track]:
        """Feed one frame of detections. Returns tracks that just appeared."""
        fresh = []
        present = set()
        self.transient = []
        for d in dets:
            if d.tid < 0:
                self.transient.append(Track(None, d, now, tracked=False))
                continue
            present.add(d.tid)
            t = self.tracks.get(d.tid)
            if t is None:
                t = Track(d.tid, d, now, tracked=True)
                self.tracks[d.tid] = t
                if d.tid not in self.seen_ids:
                    self.seen_ids.add(d.tid)
                    fresh.append(t)
            else:
                t.box += SMOOTHING * (np.asarray(d.box, np.float32) - t.box)
                t.conf = t.conf * 0.7 + d.conf * 0.3
                t.cls = d.cls
                if d.dist is None or t.dist is None:
                    t.dist = d.dist
                else:
                    t.dist = t.dist * 0.75 + d.dist * 0.25
            t.last_seen = now
            t.alive = True
            t.trail.append(t.center)

        for key in list(self.tracks):
            t = self.tracks[key]
            if key not in present:
                t.alive = False
                if now - t.last_seen > LINGER_S:
                    del self.tracks[key]
        return fresh

    def visible(self, now: float):
        """Yield (track, opacity, age_seconds)."""
        for t in self.tracks.values():
            fade = 1.0 if t.alive else max(0.0, 1.0 - (now - t.last_seen) / LINGER_S)
            yield t, fade, now - t.first_seen
        for t in self.transient:
            yield t, 1.0, 10.0

    def alive_tracks(self):
        return [t for t in self.tracks.values() if t.alive] + self.transient

    def counts(self) -> Counter:
        return Counter(t.cls for t in self.alive_tracks())


def _inside(inner: Track, outer: Track) -> bool:
    cx, cy = inner.center
    b = outer.box
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and inner.area() < 0.35 * outer.area()


def describe_scene(tracks: list[Track], labels: dict[int, str], person_cls: int | None) -> str:
    text = _describe(tracks, labels, person_cls)
    ranged = [t for t in tracks if t.dist is not None]
    if ranged:
        near = min(ranged, key=lambda t: t.dist)
        text += f"   ·   nearest: {labels.get(near.cls, '?')} {near.dist:.1f} m"
    return text


def _describe(tracks: list[Track], labels: dict[int, str], person_cls: int | None) -> str:
    """One-line live caption, e.g. '1 person holding a Phone · 2 Cups · Laptop'."""
    if not tracks:
        return "Scanning the scene…"
    people = [t for t in tracks if t.cls == person_cls]
    things = [t for t in tracks if t.cls != person_cls]
    held = [t for t in things if any(_inside(t, p) for p in people)]
    loose = Counter(t.cls for t in things if t not in held)

    parts = []
    if people:
        who = "1 person" if len(people) == 1 else f"{len(people)} people"
        if held:
            names = sorted({labels.get(t.cls, "?") for t in held})
            who += " with " + " & ".join(names)
        parts.append(who)
    for cls, n in loose.most_common():
        name = labels.get(cls, "?")
        parts.append(f"{n} {name}s" if n > 1 else name)
    return "  ·  ".join(parts)
