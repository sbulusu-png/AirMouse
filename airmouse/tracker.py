"""Hand landmark extraction with the MediaPipe Tasks HandLandmarker."""

import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import MODEL_URL, Config

# Landmark indices used across the project.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

HAND_CONNECTIONS = tuple(
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


@dataclass(frozen=True)
class Hand:
    landmarks: np.ndarray  # (21, 3): x, y normalized to 0..1 of the frame; z relative depth
    handedness: str        # the user's real hand: "Left" / "Right"
    score: float
    world: np.ndarray | None = None  # (21, 3) metric 3D joints (metres, hand-centred)


# MediaPipe labels handedness for an un-mirrored image; our frames are mirrored,
# so its labels come out swapped.
_UNMIRROR = {"Left": "Right", "Right": "Left"}


@contextmanager
def _quiet_native_stderr() -> Generator[None]:
    """Hide MediaPipe's C++ startup log spam, which env vars can't silence.

    Only wraps model creation and the first frame, so later errors stay visible.
    Does nothing when there is no console (pythonw / windowed .exe), where fd 2 is invalid.
    """
    if sys.stderr is not None:
        sys.stderr.flush()
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class HandTracker:
    def __init__(self, cfg: Config) -> None:
        if not cfg.model_path.exists():
            raise FileNotFoundError(
                f"Hand model missing at {cfg.model_path}. Download it from {MODEL_URL}"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cfg.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.detection_confidence,
            min_hand_presence_confidence=cfg.presence_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        with _quiet_native_stderr():
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._warmed_up = False

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        """Detect hands in an already-mirrored BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        if self._warmed_up:
            result = self._landmarker.detect_for_video(image, ts_ms)
        else:
            with _quiet_native_stderr():
                result = self._landmarker.detect_for_video(image, ts_ms)
            self._warmed_up = True

        return [
            Hand(
                landmarks=np.array([(p.x, p.y, p.z) for p in points], dtype=np.float32),
                handedness=_UNMIRROR.get(categories[0].category_name or "", "Unknown"),
                score=categories[0].score or 0.0,
                world=np.array([(p.x or 0.0, p.y or 0.0, p.z or 0.0) for p in world],
                               dtype=np.float32),
            )
            for points, categories, world in zip(result.hand_landmarks, result.handedness,
                                                 result.hand_world_landmarks)
        ]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
