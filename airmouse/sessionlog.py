"""Per-frame CSV log of landmarks and gesture output, for tuning thresholds offline.

Only numbers are written (no video). One file per run in logs/.
"""

import csv
import time
from pathlib import Path

from .config import LOG_DIR
from .gestures import GestureOutput
from .tracker import Hand


class SessionLog:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = LOG_DIR / time.strftime("session-%Y%m%d-%H%M%S.csv")
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        landmark_cols = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
        self._writer.writerow(["t", "mode", "events", "scroll", "left_push", "right_push",
                               *landmark_cols])

    def write(self, now: float, hand: Hand | None, out: GestureOutput | None) -> None:
        if out is None:
            return
        landmarks = [f"{v:.5f}" for v in hand.landmarks.ravel()] if hand is not None else []
        self._writer.writerow([
            f"{now:.4f}", out.mode.name, "|".join(e.name for e in out.events), out.scroll,
            f"{out.left_push:.4f}", f"{out.right_push:.4f}", *landmarks,
        ])

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
