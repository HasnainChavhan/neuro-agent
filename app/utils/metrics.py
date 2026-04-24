"""
NeuroAgent — Task Metrics Tracker
Tracks completion rates, latencies, tool usage, and retry events.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskMetrics:
    """Holds metrics for a single research task execution."""

    task_id: str
    query: str
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    success: bool = False
    sub_queries: List[str] = field(default_factory=list)
    tools_invoked: List[str] = field(default_factory=list)
    retry_events: int = 0
    fallback_events: int = 0
    tokens_used: int = 0

    @property
    def latency_ms(self) -> float:
        end = self.finished_at or time.monotonic()
        return (end - self.started_at) * 1000

    def finish(self, success: bool = True) -> None:
        self.finished_at = time.monotonic()
        self.success = success


class MetricsStore:
    """In-memory metrics store; swap for Prometheus/StatsD in production."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskMetrics] = {}
        self._counters: Dict[str, int] = defaultdict(int)

    # ── Task lifecycle ─────────────────────────────────────────────────────

    def start_task(self, task_id: str, query: str) -> TaskMetrics:
        m = TaskMetrics(task_id=task_id, query=query)
        self._tasks[task_id] = m
        self._counters["tasks_started"] += 1
        return m

    def finish_task(self, task_id: str, success: bool = True) -> None:
        if task := self._tasks.get(task_id):
            task.finish(success)
            key = "tasks_succeeded" if success else "tasks_failed"
            self._counters[key] += 1

    def get_task(self, task_id: str) -> TaskMetrics | None:
        return self._tasks.get(task_id)

    # ── Aggregates ──────────────────────────────────────────────────────────

    def completion_rate(self) -> float:
        started = self._counters["tasks_started"]
        if started == 0:
            return 0.0
        return self._counters["tasks_succeeded"] / started

    def summary(self) -> dict:
        return {
            "tasks_started": self._counters["tasks_started"],
            "tasks_succeeded": self._counters["tasks_succeeded"],
            "tasks_failed": self._counters["tasks_failed"],
            "completion_rate": round(self.completion_rate() * 100, 2),
            "total_retries": self._counters["retries"],
            "total_fallbacks": self._counters["fallbacks"],
        }

    def record_retry(self) -> None:
        self._counters["retries"] += 1

    def record_fallback(self) -> None:
        self._counters["fallbacks"] += 1


# Global singleton
metrics_store = MetricsStore()
