"""Wall-clock budget shared by every stage of a run."""
from __future__ import annotations

import time

# Measured from the moment the package is imported, so interpreter start-up
# and library imports count against the limit as they will for the grader.
PROCESS_START = time.perf_counter()


class Budget:
    def __init__(self, seconds: float, start: float | None = None):
        self.total = float(seconds)
        self.start = PROCESS_START if start is None else start

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def left(self) -> float:
        return self.total - self.elapsed()

    def fraction_left(self) -> float:
        return self.left() / self.total
