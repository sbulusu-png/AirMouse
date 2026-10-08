"""Drawing helpers for the hand skeleton and status HUD."""

import cv2
import numpy as np

from .tracker import FINGERTIPS, HAND_CONNECTIONS, Hand

BONE_COLOR = (255, 200, 80)   # BGR
JOINT_COLOR = (255, 255, 255)
TIP_COLOR = (80, 220, 255)
TEXT_COLOR = (240, 240, 240)
ACTIVE_COLOR = (120, 230, 120)
IDLE_COLOR = (130, 130, 130)


def to_pixels(hand: Hand, width: int, height: int) -> np.ndarray:
    """Normalized (0..1) landmarks -> integer frame pixel coordinates, shape (21, 2)."""
    return (hand.landmarks[:, :2] * (width, height)).astype(np.int32)


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    h, w = frame.shape[:2]
    pts = to_pixels(hand, w, h)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), BONE_COLOR, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        is_tip = i in FINGERTIPS
        cv2.circle(frame, tuple(p), 7 if is_tip else 4,
                   TIP_COLOR if is_tip else JOINT_COLOR, -1, cv2.LINE_AA)


def draw_active_region(frame: np.ndarray, rect: tuple[int, int, int, int], active: bool) -> None:
    color = ACTIVE_COLOR if active else IDLE_COLOR
    cv2.rectangle(frame, rect[:2], rect[2:], color, 2, cv2.LINE_AA)


def draw_pointer(frame: np.ndarray, hand: Hand, index: int) -> None:
    """Ring the landmark that drives the cursor."""
    h, w = frame.shape[:2]
    p = tuple(to_pixels(hand, w, h)[index])
    cv2.circle(frame, p, 14, ACTIVE_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    TEXT_COLOR, 1, cv2.LINE_AA)
