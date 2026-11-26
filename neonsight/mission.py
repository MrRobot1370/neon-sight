"""Mission mode: a short 'show me a ___' game for live demos.

A mission is a fixed number of rounds (config: mission.rounds). Each round asks for one object;
the player shows it to the camera, skips it, or runs out of time. At the end a results card
lists every round.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

HOLD_S = 0.4  # target must be seen continuously this long
RESULT_S = 2.4  # how long the found / time-up banner stays before the next round
SKIP_RESULT_S = 0.9  # a skip moves on quickly
RESULTS_CARD_S = 15.0  # how long the final results card stays on screen


@dataclass
class RoundResult:
    target: int
    outcome: str  # found | timeout | skipped
    secs: float | None = None
    points: int = 0


def make_plan(pool: list[int], rounds: int) -> list[int]:
    """Random order, every object once before any repeats, never the same object twice in a row."""
    plan: list[int] = []
    while len(plan) < rounds:
        batch = random.sample(pool, len(pool))
        if plan and len(batch) > 1 and batch[0] == plan[-1]:
            batch[0], batch[-1] = batch[-1], batch[0]
        plan += batch
    return plan[:rounds]


class Mission:
    def __init__(self, rounds: int = 5, seconds: float = 15.0):
        self.rounds = rounds
        self.seconds = seconds
        self.active = False  # a round is running (hunt or the short result banner after it)
        self.state = "idle"  # idle | hunt | found | timeout | skipped | done
        self.plan: list[int] = []
        self.results: list[RoundResult] = []
        self.index = 0
        self.started = 0.0
        self.finished_at = 0.0
        self._hold_from: float | None = None
        self._until = 0.0

    # ------------------------------------------------------------ queries
    @property
    def target(self) -> int | None:
        return self.plan[self.index] if self.active and self.index < len(self.plan) else None

    @property
    def round(self) -> int:
        return self.index + 1

    @property
    def total(self) -> int:
        return len(self.plan)

    @property
    def score(self) -> int:
        return sum(r.points for r in self.results)

    @property
    def found(self) -> int:
        return sum(r.outcome == "found" for r in self.results)

    @property
    def best(self) -> float | None:
        times = [r.secs for r in self.results if r.outcome == "found"]
        return min(times) if times else None

    @property
    def last(self) -> RoundResult | None:
        return self.results[-1] if self.results else None

    def time_left(self, now: float) -> float:
        return max(0.0, self.seconds - (now - self.started))

    def showing_results(self, now: float) -> bool:
        return self.state == "done" and now - self.finished_at < RESULTS_CARD_S

    def dismiss_results(self):
        """Close the results card early (the sidebar keeps the final score)."""
        self.finished_at = -RESULTS_CARD_S

    # ------------------------------------------------------------ control
    def start(self, pool: list[int], now: float) -> bool:
        if not pool:
            return False
        self.plan = make_plan(pool, self.rounds)
        self.results = []
        self.index = 0
        self.active = True
        self._begin(now)
        return True

    def stop(self):
        self.active = False
        self.state = "idle"

    def skip(self, now: float):
        """Player doesn't have the object: record it and move on."""
        if self.active and self.state == "hunt":
            self._finish_round("skipped", now, SKIP_RESULT_S)
            return ("skipped", None)
        return None

    def _begin(self, now: float):
        self.state = "hunt"
        self.started = now
        self._hold_from = None

    def _finish_round(self, outcome: str, now: float, show_s: float, secs: float | None = None, points: int = 0):
        self.results.append(RoundResult(self.plan[self.index], outcome, secs, points))
        self.state = outcome
        self._until = now + show_s

    def update(self, tracks, valid: set[int], now: float):
        """Returns ('found', track) | ('timeout'|'skipped'|'next'|'done', None) | None.

        `valid` holds the targets that can still be detected (a typed word may have been cleared).
        """
        if not self.active:
            return None
        if self.state == "hunt":
            target = self.plan[self.index]
            if target not in valid:
                return self.skip(now)
            hits = [t for t in tracks if t.cls == target]
            if hits:
                if self._hold_from is None:
                    self._hold_from = now
                elif now - self._hold_from >= HOLD_S:
                    secs = now - self.started
                    points = max(10, round(100 - 90 * secs / self.seconds))
                    self._finish_round("found", now, RESULT_S, secs, points)
                    return ("found", max(hits, key=lambda t: t.conf))
            else:
                self._hold_from = None
            if now - self.started > self.seconds:
                self._finish_round("timeout", now, RESULT_S)
                return ("timeout", None)
        elif now >= self._until:
            if self.index + 1 < len(self.plan):
                self.index += 1
                self._begin(now)
                return ("next", None)
            self.active = False
            self.state = "done"
            self.finished_at = now
            return ("done", None)
        return None
