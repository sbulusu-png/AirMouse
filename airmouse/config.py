"""Tunable parameters for the whole pipeline."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Packaged as AirMouse.exe (PyInstaller), bundled files live in sys._MEIPASS and the
# install folder may be read-only; from source, everything sits in the project folder.
FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_ROOT = (Path(getattr(sys, "_MEIPASS", "")) if FROZEN
                else Path(__file__).resolve().parent.parent)      # bundled resources
USER_DIR = Path(os.environ.get("APPDATA", Path.home())) / "AirMouse"  # settings, error log
LOG_DIR = USER_DIR / "logs" if FROZEN else PROJECT_ROOT / "logs"   # recordings, session logs
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class Config:
    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # Hand detection (MediaPipe)
    model_path: Path = PROJECT_ROOT / "models" / "hand_landmarker.task"
    max_hands: int = 1
    detection_confidence: float = 0.7
    presence_confidence: float = 0.7
    tracking_confidence: float = 0.7

    # Cursor mapping (B2)
    cursor_speed: float = 2.2        # screen widths per frame width of hand motion (sensitivity)
    area_center_y: float = 0.36      # vertical centre of the hand's control area (0 top, 1 bottom)
    area_center_x: float = 0.5       # horizontal centre of the control area (0 left, 1 right)
    area_height: float = 0.44        # control area height, fraction of the camera view. Measured:
                                     # this user's aim spans 0.14-0.60 of the view; the old
                                     # 0.17-0.67 box's bottom was reached in 0.2% of frames
    taskbar_zone: float = 0.15       # share of the hand area given to the taskbar strip (0 = off)
    smoothing_alpha: float = 0.25    # EMA weight of the newest sample, 0..1
    fingertip_follow: float = 0.0    # 0 = cursor follows the index knuckle (steady during
                                     # gestures), 1 = the raw fingertip (wobbles with them)

    # Gestures (B3)
    push_threshold: float = 0.35     # finger-push sensitivity: relative drop from rest (lower = easier)
                                     # (real taps peak 0.3-1.25; reaching drifts hit 0.34-0.41)
    click_cooldown: float = 0.3      # seconds between clicks (short enough for double-click)
    double_click_window: float = 1.5  # a 2nd left click on the same spot this soon = double click
    scroll_speed: float = 1500.0     # wheel units per palm length of hand motion (120 = 1 notch)
    click_lookback: float = 0.2      # on a push, snap cursor to where it was this long ago
    drag_deadzone: float = 30.0      # screen px of hand motion while pressed before a drag starts
    release_hold: float = 0.15       # keep cursor pinned this long after a press ends

    # Debugging: per-frame landmark log written to logs/ (no video is recorded)
    session_log: bool = False
