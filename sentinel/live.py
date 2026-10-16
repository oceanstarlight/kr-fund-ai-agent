"""In-app live GDELT tracker shared by every user session of the app.

The app process keeps its own rolling history of the 15-minute GDELT files and runs the same
detector as the scheduled monitor (``scripts/monitor.py``), so the map updates every tick
without reading the Actions cache. Only aggregated per-country rows are kept in memory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from sentinel import gdelt_monitor as g

WARM_SLOTS = 96  # one day of files on first start (enough for GDELT_MIN_HISTORY)
MAX_CATCHUP = 16
MAX_ALERTS_KEPT = 20


@dataclass
class LiveSnapshot:
    slot: pd.Timestamp | None
    scores: pd.DataFrame  # g.country_scores output for the latest slot
    alerts: list[g.GdeltAlert] = field(default_factory=list)  # newest first, cooldown applied
    history_slots: int = 0
    error: str = ""


class LiveTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.history = pd.DataFrame()
        self.last_slot: pd.Timestamp | None = None
        self.streak: dict[str, int] = {}
        self.last_alerts: dict[str, str] = {}
        self.alerts: list[g.GdeltAlert] = []
        self.scores = pd.DataFrame()
        self.error = ""

    def snapshot(self) -> LiveSnapshot:
        n = self.history["slot"].nunique() if len(self.history) else 0
        return LiveSnapshot(self.last_slot, self.scores.copy(), list(self.alerts), n, self.error)

    def update(self) -> LiveSnapshot:
        """Process every new 15-minute file since the last call. Never raises."""
        with self._lock:  # one download per tick, whichever session asks first
            try:
                self._update()
                self.error = ""
            except Exception as exc:  # noqa: BLE001 - the map must keep the last good state
                self.error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return self.snapshot()

    def _update(self) -> None:
        latest = g.latest_slot()
        if self.last_slot is not None and latest <= self.last_slot:
            return
        if self.last_slot is None:
            warm = [latest - i * g.SLOT for i in range(WARM_SLOTS, 0, -1)]
            data = g.fetch_slots(warm)
            if data:
                self.history = pd.concat([g.aggregate(data[s]) for s in sorted(data)], ignore_index=True)
            todo = [latest]
        else:
            n = int((latest - self.last_slot) / g.SLOT)
            todo = [self.last_slot + i * g.SLOT for i in range(max(1, n - MAX_CATCHUP + 1), n + 1)]
        data = g.fetch_slots(todo)
        for s in sorted(data):
            agg = g.aggregate(data[s])
            self.scores = g.country_scores(self.history, agg)
            found, self.streak = g.detect(self.history, agg, data[s], self.last_alerts, self.streak)
            for a in found:
                self.last_alerts[a.country] = a.slot
            self.alerts = (found + self.alerts)[:MAX_ALERTS_KEPT]
            self.history = pd.concat([self.history, agg], ignore_index=True)
        self.last_slot = latest
        if len(self.history):
            self.history = g.trim_history(self.history, latest)
