"""Durable child-Operator reservations under the existing campaign lease.

Model request attempts remain authoritative in ModelGateway's shared journal.
This journal owns Operator attempts, input rows, output bytes and one deadline.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .state_store import atomic_write_json


class DataBudgetExceeded(RuntimeError):
    pass


class DataBudgetJournal:
    def __init__(self, path: Path, budget, binding: str, *, now=time.time):
        self.path, self.budget, self.binding, self.now = path, budget, binding, now
        if path.exists():
            self.data = json.loads(path.read_text())
            if self.data.get("binding") != binding or self.data.get("budget") != budget.to_dict():
                raise ValueError("data budget binding changed")
            self._validate()
        else:
            self.data = {"binding": binding, "budget": budget.to_dict(), "started_at": now(),
                         "attempts": {}, "rows": {}, "files": {}}
            self.save()

    def _validate(self):
        for name in ("attempts", "rows", "files"):
            if not isinstance(self.data.get(name), dict) or any(
                    type(n) is not int or n < 0 for n in self.data[name].values()):
                raise ValueError("invalid data budget journal")
        if (type(self.data.get("started_at")) not in (int, float) or not math.isfinite(self.data["started_at"])
                or self.data["started_at"] > self.now()):
            raise ValueError("invalid data budget deadline")
        for key, maximum in (("attempts", self.budget.max_operator_calls),
                             ("rows", self.budget.max_input_rows),
                             ("files", self.budget.max_output_bytes)):
            if sum(self.data[key].values()) > maximum:
                raise DataBudgetExceeded(f"persisted {key} exceeds approved budget")

    def save(self):
        atomic_write_json(self.path, self.data)

    def remaining_seconds(self):
        return self.budget.max_wall_seconds - (self.now() - self.data["started_at"])

    def check(self):
        if self.remaining_seconds() <= 0:
            raise DataBudgetExceeded("data run wall-time budget exhausted")

    def reserve_operator(self, node_id):
        self.check()
        if sum(self.data["attempts"].values()) >= self.budget.max_operator_calls:
            raise DataBudgetExceeded("data Operator-call budget exhausted")
        self.data["attempts"][node_id] = self.data["attempts"].get(node_id, 0) + 1
        self.save()

    def reserve_rows(self, key, count):
        self._reserve("rows", key, count, self.budget.max_input_rows)

    def reserve_file(self, key, size):
        self._reserve("files", key, size, self.budget.max_output_bytes)

    def _reserve(self, category, key, count, maximum):
        self.check()
        if type(count) is not int or count < 0:
            raise ValueError("budget reservation must be a non-negative integer")
        old = self.data[category].get(key)
        if old is not None:
            if old != count:
                raise ValueError("budget reservation identity changed")
            return
        if sum(self.data[category].values()) + count > maximum:
            raise DataBudgetExceeded(f"data {category} budget exhausted")
        self.data[category][key] = count
        self.save()

    def summary(self):
        return {"operator_calls": sum(self.data["attempts"].values()),
                "input_rows": sum(self.data["rows"].values()),
                "output_bytes": sum(self.data["files"].values()),
                "wall_seconds": self.now() - self.data["started_at"],
                "limits": self.budget.to_dict()}
