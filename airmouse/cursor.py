"""Hand position -> screen cursor: active-region mapping, EMA smoothing, OS cursor control."""

import ctypes
import ctypes.wintypes
from collections import deque

import numpy as np
import pyautogui

pyautogui.FAILSAFE = True  # slam the physical mouse into a screen corner to abort

MOUSEEVENTF_LEFTUP = 0x0004


def system_double_click_time() -> float:
    """Windows' double-click interval in seconds (500 ms by default)."""
    try:
        return ctypes.windll.user32.GetDoubleClickTime() / 1000.0
    except (AttributeError, OSError):
        return 0.5


Region = tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels
Bands = tuple[int, int, int, int]           # taskbar thickness at left, top, right, bottom (px)
NO_BANDS: Bands = (0, 0, 0, 0)
SPI_GETWORKAREA = 0x0030


def taskbar_bands(screen_size: tuple[int, int]) -> Bands:
    """Where the taskbar is: the gap between the screen and Windows' work area on each edge
    (e.g. (0, 0, 0, 60) for a 60 px taskbar at the bottom). An auto-hidden taskbar gives
    no gap, so no band."""
    rect = ctypes.wintypes.RECT()
    try:
        if not ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0,
                                                          ctypes.byref(rect), 0):
            return NO_BANDS
    except (AttributeError, OSError):
        return NO_BANDS
    sw, sh = screen_size
    bands = (rect.left, rect.top, sw - rect.right, sh - rect.bottom)
    # Ignore nonsense (e.g. a work area on another monitor): bands are thin strips.
    return tuple(b if 0 <= b < min(sw, sh) / 4 else 0 for b in bands)  # type: ignore[return-value]


def _map_axis(p: float, r0: float, r1: float, length: int, near: int, far: int,
              zone: float) -> float:
    """One axis of map_to_screen: the hand's position across the region -> screen px.

    A taskbar band at either end (`near`/`far` px thick) gets `zone` of the hand's travel
    instead of its tiny share of the screen: easier to reach and to stay on. The rest of
    the screen shares what's left, linearly. With no bands this is a plain linear map."""
    u = min(max((p - r0) / (r1 - r0), 0.0), 1.0)
    hand = [0.0]
    screen = [1.0]
    if near > 0 and zone > 0:
        hand.append(zone)
        screen.append(float(near))
    if far > 0 and zone > 0:
        hand.append(1.0 - zone)
        screen.append(float(length - far))
    hand.append(1.0)
    screen.append(float(length - 2))
    return float(np.interp(u, hand, screen))


def active_region(frame_size: tuple[int, int], screen_size: tuple[int, int],
                  speed: float, center_y: float, height: float | None = None,
                  center_x: float = 0.5) -> Region:
    """The rectangle of the camera frame that maps onto the whole screen.

    Its width is frame_width / speed. Its height is `height` (fraction of the frame
    height), or, if None, the screen's aspect ratio (equal cursor speed both ways). A hand
    can't go as low as it can go high (the wrist leaves the view), so a shorter box keeps
    the bottom reachable. Centred at (`center_x`, `center_y`), kept inside the frame.
    """
    fw, fh = frame_size
    sw, sh = screen_size
    w = fw / max(speed, 1.0)
    h = min(w * sh / sw if height is None else height * fh, fh)
    cx = min(max(center_x * fw, w / 2), fw - w / 2)
    cy = min(max(center_y * fh, h / 2), fh - h / 2)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def map_to_screen(
    norm_xy: np.ndarray,
    frame_size: tuple[int, int],
    region: Region,
    screen_size: tuple[int, int],
    bands: Bands = NO_BANDS,
    taskbar_zone: float = 0.0,
) -> np.ndarray:
    """Map a normalized (0..1) landmark position to screen pixels.

    Positions inside `region` map onto the full screen (with taskbar bands enlarged, see
    _map_axis); positions outside it clamp to the screen edge. The result stays 1 px inside
    the screen so hand motion can never land on a PyAutoGUI failsafe corner.
    """
    fw, fh = frame_size
    sw, sh = screen_size
    x0, y0, x1, y1 = region
    left, top, right, bottom = bands
    sx = _map_axis(norm_xy[0] * fw, x0, x1, sw, left, right, taskbar_zone)
    sy = _map_axis(norm_xy[1] * fh, y0, y1, sh, top, bottom, taskbar_zone)
    return np.array((sx, sy), dtype=np.float64)


class EmaFilter:
    """Exponential moving average: out = alpha * new + (1 - alpha) * out."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._value: np.ndarray | None = None

    def update(self, sample: np.ndarray) -> np.ndarray:
        prev = self._value
        value = sample.astype(np.float64) if prev is None else \
            self.alpha * sample + (1.0 - self.alpha) * prev
        self._value = value
        return value

    def reset(self) -> None:
        """Forget history, so the next sample is taken as-is (e.g. after the hand is lost)."""
        self._value = None


class ClickStabilizer:
    """Keeps the cursor where you aimed while a finger push holds a mouse button.

    Pushing a finger toward the camera foreshortens it, moving the fingertip (the normal
    cursor driver) by up to ~100 screen px. So on a press the cursor snaps back to where
    it was `lookback` seconds earlier, and while the button is held it moves only with
    `ref`: a point that follows the whole hand but not the finger bend (the index
    knuckle). Once `ref` has moved more than `drag_deadzone` px the press becomes a drag
    and the cursor follows the hand; a `grab` (fist) follows the hand immediately.
    After release the cursor stays put for `release_hold` seconds, then glides back to the
    fingertip over `HANDOVER_S` instead of jumping (after a fist the fingertip can be
    hundreds of px away from where the hand was carrying the cursor).
    """

    HANDOVER_S = 0.35

    def __init__(self, lookback: float, drag_deadzone: float, release_hold: float) -> None:
        self.lookback = lookback
        self.drag_deadzone = drag_deadzone
        self.release_hold = release_hold
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._anchor: np.ndarray | None = None      # set while a press is being stabilized
        self._ref_origin = np.zeros(2)
        self._dragging = False
        self._out = np.zeros(2)
        self._released_at: float | None = None
        self._handover: tuple[float, np.ndarray] | None = None  # (start time, offset)

    def update(self, xy: np.ndarray, ref: np.ndarray, now: float, pressed: bool, held: bool,
               draggable: bool, grab: bool = False) -> np.ndarray:
        """xy: smoothed fingertip target. ref: smoothed knuckle target (screen px)."""
        self._history.append((now, xy))
        while self._history and self._history[0][0] < now - self.lookback - 0.5:
            self._history.popleft()

        if pressed:
            past = [p for t, p in self._history if t <= now - self.lookback]
            self._anchor = past[-1] if past else self._history[0][1]
            self._ref_origin = ref
            self._dragging = False
            self._released_at = None
            self._handover = None

        anchor = self._anchor
        if anchor is None:
            return self._blend(xy, now)
        if held:
            offset = ref - self._ref_origin
            if grab or (draggable and np.linalg.norm(offset) > self.drag_deadzone):
                self._dragging = True
            self._out = anchor + offset if self._dragging else anchor
        else:
            if self._released_at is None:
                self._released_at = now
            if now - self._released_at >= self.release_hold:
                self._anchor = None
                self._handover = (now, self._out - xy)
                return self._blend(xy, now)
        return self._out

    def _blend(self, xy: np.ndarray, now: float) -> np.ndarray:
        if self._handover is None:
            return xy
        start, offset = self._handover
        remaining = 1.0 - (now - start) / self.HANDOVER_S
        if remaining <= 0:
            self._handover = None
            return xy
        return xy + offset * remaining

    def reset(self) -> None:
        self._history.clear()
        self._anchor = None
        self._dragging = False
        self._released_at = None
        self._handover = None


class CursorController:
    def __init__(self) -> None:
        w, h = pyautogui.size()
        self.screen_size = (w, h)
        self._left_held = False
        self._bands = NO_BANDS
        self._bands_at = -1e9

    def bands(self, now: float) -> Bands:
        """The taskbar bands, re-read every few seconds (the taskbar can move or hide)."""
        if now - self._bands_at > 3.0:
            self._bands = taskbar_bands(self.screen_size)
            self._bands_at = now
        return self._bands

    # _pause=False skips PyAutoGUI's default 0.1 s post-call sleep, which would cap us at
    # 10 fps. The failsafe check still runs and raises FailSafeException.

    def move(self, xy: np.ndarray) -> None:
        pyautogui.moveTo(int(round(xy[0])), int(round(xy[1])), _pause=False)

    def left_down(self) -> None:
        pyautogui.mouseDown(button="left", _pause=False)
        self._left_held = True

    def left_up(self) -> None:
        if self._left_held:
            self._left_held = False
            pyautogui.mouseUp(button="left", _pause=False)

    def left_click(self) -> None:
        pyautogui.click(button="left", _pause=False)

    def double_click(self) -> None:
        pyautogui.doubleClick(button="left", _pause=False)

    def right_click(self) -> None:
        pyautogui.click(button="right", _pause=False)

    def scroll(self, amount: int) -> None:
        if amount:
            pyautogui.scroll(amount, _pause=False)

    def release(self) -> None:
        """Never leave the left button stuck down, even when the failsafe has fired."""
        if not self._left_held:
            return
        try:
            self.left_up()
        except pyautogui.FailSafeException:
            # The failsafe blocks PyAutoGUI calls in a corner; release via Win32 directly.
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
