"""Synthetic hands for tests: an upright hand, palm to the camera, in a 1280x720 frame."""

from collections.abc import Sequence

import numpy as np

from airmouse.gestures import Event, GestureEngine, GestureOutput
from airmouse.tracker import Hand

FS = (1280, 720)
DT = 1 / 30

_MCP = {"i": (600, 455), "m": (640, 450), "r": (680, 455), "p": (715, 470)}
_BASE = {"i": 5, "m": 9, "r": 13, "p": 17}
_SEGMENTS = np.array([50.0, 85.0, 115.0])   # PIP, DIP, tip distance above the knuckle


def make_hand(ext: tuple[str, ...] = ("i", "m", "r", "p"), push: str | None = None,
              amount: float = 0.0, dy: float = 0.0, tilt: float = 1.0,
              dx: float = 0.0, thumb_fold: float = 0.0) -> Hand:
    """ext: straight fingers (others curled). push: finger(s) tipped toward the camera,
    e.g. "i" or "mr", by `amount` (0..1 of 90 degrees). dx/dy: shift in px. tilt: vertical squash factor
    simulating the palm tilting back. thumb_fold: 0 = thumb out, 1 = folded across the
    palm to the base of the little finger."""
    p = np.zeros((21, 2))
    z = np.zeros(21)
    p[0] = (640, 600)
    for f, (x, y) in _MCP.items():
        if f == "m" and set(ext) == {"i", "m"}:
            x = 640.0  # scroll pose: index & middle side by side
        b = _BASE[f]
        p[b] = (x, y)
        if f not in ext:
            p[b + 1:b + 4] = [(x, y - 40), (x, y - 10), (x, y + 20)]
        elif push is not None and f in push:
            ang = amount * np.pi / 2
            p[b + 1:b + 4] = [(x, y - s * np.cos(ang)) for s in _SEGMENTS]
            z[b + 1:b + 4] = -_SEGMENTS * np.sin(ang) / FS[0]
        else:
            p[b + 1:b + 4] = [(x, y - s) for s in _SEGMENTS]
    p[1:5] = [(590, 560), (560, 520), (540, 490), (525, 465)]
    fold_to = np.array([700.0, 500.0])
    for joint, weight in ((2, 0.3), (3, 0.65), (4, 1.0)):
        p[joint] += (fold_to - p[joint]) * weight * thumb_fold
    p[:, 1] = 600 + (p[:, 1] - 600) * tilt + dy
    p[:, 0] += dx
    lm = np.zeros((21, 3), np.float32)
    lm[:, :2] = p / FS
    lm[:, 2] = z
    return Hand(lm, "Right", 0.99)


REST = make_hand()


def fold(frames_in: int = 4) -> list[Hand]:
    """Fold the thumb into the palm over 3 frames, hold, and bring it back out."""
    ramp = [make_hand(thumb_fold=k / 3) for k in range(1, 4)]
    return ramp + [make_hand(thumb_fold=1.0)] * frames_in + ramp[::-1]


def tap(finger: str, frames_down: int = 4, amount: float = 0.5) -> list[Hand]:
    """Push a finger forward over 3 frames, hold, and pull it back over 3 frames."""
    ramp = [make_hand(push=finger, amount=amount * k / 3) for k in range(1, 4)]
    return ramp + [make_hand(push=finger, amount=amount)] * frames_down + ramp[::-1]


_TOUCH_POINT = np.array([620.0, 395.0, -77.0])   # in front of the palm, in px (z ~ x scale)


def make_touch(closed: float = 1.0) -> Hand:
    """Open hand whose five fingertips converge on one point in front of the palm.
    closed: 0 = open hand, 1 = all fingertips touching."""
    lm = REST.landmarks.copy()
    p3 = np.column_stack([lm[:, :2] * FS, lm[:, 2] * FS[0]])
    for chain in ((2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)):
        for joint, weight in zip(chain, (0.4, 0.7, 1.0)):     # tip moves all the way
            p3[joint] += (_TOUCH_POINT - p3[joint]) * weight * closed
    lm[:, :2] = p3[:, :2] / FS
    lm[:, 2] = p3[:, 2] / FS[0]
    return Hand(lm.astype(np.float32), "Right", 0.99)


def touch(frames_closed: int = 4) -> list[Hand]:
    """Bring all five fingertips together over 3 frames, hold, and open over 3 frames."""
    ramp = [make_touch(k / 3) for k in range(1, 4)]
    return ramp + [make_touch(1.0)] * frames_closed + ramp[::-1]


def run(engine: GestureEngine, frames: Sequence[Hand | None],
        t0: float = 0.0) -> tuple[list[GestureOutput], float]:
    t, outs = t0, []
    for h in frames:
        outs.append(engine.update(h, FS, t))
        t += DT
    return outs, t


def events(outs: list[GestureOutput]) -> list[Event]:
    return [e for o in outs for e in o.events]


def make_pinch(closed: float = 1.0, others_out: bool = True) -> Hand:
    """Thumb and index tips meeting in front of the palm (the double-click pinch).
    others_out: middle/ring/pinky stay extended (else they curl in, like a fist)."""
    base = REST if others_out else make_hand(ext=("i",))
    lm = base.landmarks.copy()
    p3 = np.column_stack([lm[:, :2] * FS, lm[:, 2] * FS[0]])
    meet = (p3[4] + p3[8]) / 2 + np.array([0.0, 0.0, -40.0])
    for chain in ((2, 3, 4), (6, 7, 8)):
        for joint, weight in zip(chain, (0.3, 0.65, 1.0)):
            p3[joint] += (meet - p3[joint]) * weight * closed
    lm[:, :2] = p3[:, :2] / FS
    lm[:, 2] = p3[:, 2] / FS[0]
    return Hand(lm.astype(np.float32), "Right", 0.99)


def pinch(frames_closed: int = 4) -> list[Hand]:
    """Close a thumb-index pinch over 3 frames, hold, and open over 3 frames."""
    ramp = [make_pinch(k / 3) for k in range(1, 4)]
    return ramp + [make_pinch(1.0)] * frames_closed + ramp[::-1]
