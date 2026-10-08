"""The capture -> track -> gesture -> mouse pipeline, running on a background thread.

Other threads talk to it only through: set_enabled(), update_config(), reset_config(),
config_dict(), snapshot(), and the on_update callback it fires once per frame.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pyautogui

from .camera import Camera
from .config import Config
from .cursor import (ClickStabilizer, CursorController, EmaFilter, Region, active_region,
                     map_to_screen, system_double_click_time)
from .gestures import Event, GestureEngine, GestureOutput, Mode
from .sessionlog import SessionLog
from .settings import PARAMS_BY_NAME, apply_values, to_dict
from .tracker import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, WRIST, Hand, HandTracker

PREVIEW_WIDTH = 640        # streamed video is downscaled to this width
JPEG_QUALITY = 70
CAMERA_RETRY_S = 2.0
CALIBRATION_S = 8.0       # "fit area to my reach": how long the user sweeps their hand
REACH_INSET = 0.9         # fit the area slightly inside the extremes reached, so the screen
                          # edges come without straining
MIN_CALIBRATION_POINTS = 60
AREA_HOME_AFTER_S = 1.5   # hand out of view this long: the sliding control area goes home
DOUBLE_CLICK_RADIUS = 0.03  # ...and within this fraction of the screen width (~60 px)

# Measured on real data: when pointing, the fingertip sits this many palm lengths above
# the index knuckle. The cursor aims there, so it lands where the finger points.
KNUCKLE_TO_TIP = 0.78


def pointer_position(hand: Hand, frame_size: tuple[int, int], fingertip_follow: float
                     ) -> np.ndarray:
    """Where the cursor aims, as a normalized frame position.

    Driven by the index knuckle, which moves with the hand but not with the fingers:
    during a touch, fist or push the fingertip moves 1.7-3x more than the knuckle, while
    for deliberate hand motion they move alike (1.1-1.2x). The knuckle is lifted by
    KNUCKLE_TO_TIP palm lengths so the cursor sits where the finger points.
    `fingertip_follow` blends toward the raw fingertip (1 = original fingertip tracking).
    """
    fw, fh = frame_size
    px = hand.landmarks[:, :2] * (fw, fh)
    palm = float(np.linalg.norm(px[WRIST] - px[MIDDLE_MCP]))
    knuckle = px[INDEX_MCP] + (0.0, -KNUCKLE_TO_TIP * palm)
    aim = knuckle + fingertip_follow * (px[INDEX_TIP] - knuckle)
    return aim / (fw, fh)


UpdateCallback = Callable[[dict[str, Any], bytes | None], None]


def fit_area(points: np.ndarray) -> dict[str, float]:
    """Control-area settings fitted to where a hand actually reached.

    `points` are normalized cursor-aim positions collected while the user swept their hand
    to the edges they can comfortably reach. Raises ValueError with advice if the sweep
    wasn't usable.
    """
    if len(points) < MIN_CALIBRATION_POINTS:
        raise ValueError("Couldn't see your hand for long enough. Keep it in view and try again.")
    x_lo, x_hi = np.percentile(points[:, 0], [3, 97])
    y_lo, y_hi = np.percentile(points[:, 1], [3, 97])
    width = (x_hi - x_lo) * REACH_INSET
    height = (y_hi - y_lo) * REACH_INSET
    if width < 0.12 or height < 0.1:
        raise ValueError("Your hand barely moved. Sweep it out to every edge you can "
                         "comfortably reach, then try again.")

    def clamp(v: float, lo: float, hi: float) -> float:
        return float(min(max(v, lo), hi))

    return {
        "cursor_speed": round(clamp(1.0 / width, 1.0, 5.0), 1),
        "area_height": round(clamp(height, 0.25, 0.8), 2),
        "area_center_x": round(clamp((x_lo + x_hi) / 2, 0.25, 0.75), 2),
        "area_center_y": round(clamp((y_lo + y_hi) / 2, 0.25, 0.75), 2),
    }


@dataclass
class Calibration:
    id: int
    started: float | None = None          # set on the first frame after the request
    points: list[np.ndarray] = field(default_factory=list)
    result: dict[str, float] | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.result is None and self.error is None


@dataclass
class Snapshot:
    frame: np.ndarray | None = None       # latest mirrored BGR frame, no overlay
    hand: Hand | None = None
    state: dict[str, Any] = field(default_factory=dict)


class Engine:
    def __init__(self, cfg: Config, enabled: bool = True,
                 on_update: UpdateCallback | None = None) -> None:
        self.cfg = cfg
        self.on_update = on_update
        self.want_jpeg = False             # set by the server while a client is watching
        self._enabled = enabled
        self._lock = threading.Lock()
        self._restart: set[str] = set()    # components to rebuild: "camera" / "tracker"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot = Snapshot()
        self._notice: str | None = None    # one-shot message for the UI, e.g. failsafe

        self.cursor = CursorController()
        self.smoother = EmaFilter(cfg.smoothing_alpha)      # fingertip: drives the cursor
        self.ref_smoother = EmaFilter(cfg.smoothing_alpha)  # knuckle: hand motion during a press
        self.stabilizer = ClickStabilizer(cfg.click_lookback, cfg.drag_deadzone,
                                          cfg.release_hold)
        self.gestures = GestureEngine(cfg)
        self._control_active = False       # engine-thread view of _enabled
        self._fps = 0.0
        self._pointer: np.ndarray | None = None        # where the cursor was last put
        self._area_offset = np.zeros(2)                 # how far the control area has slid (px)
        self._calibration: Calibration | None = None
        self._calibration_count = 0
        self._hand_lost_since: float | None = None
        self._last_click: tuple[float, np.ndarray] | None = None  # for double clicks
        self._double_click_time = system_double_click_time()
        self._last_frame_t = time.perf_counter()

    # ---- public API (any thread) -------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="airmouse-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled  # the engine thread releases buttons when it sees this

    def update_config(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and apply settings; returns what changed. Raises ValueError if invalid."""
        with self._lock:
            changed = apply_values(self.cfg, values)
            self._after_config_change(changed)
        return changed

    def reset_config(self) -> dict[str, Any]:
        return self.update_config(to_dict(Config()))

    def config_dict(self) -> dict[str, Any]:
        with self._lock:
            return to_dict(self.cfg)

    def start_calibration(self) -> None:
        """Measure the user's reach for CALIBRATION_S; mouse control pauses meanwhile."""
        with self._lock:
            self._calibration_count += 1
            self._calibration = Calibration(self._calibration_count)

    def cancel_calibration(self) -> None:
        with self._lock:
            self._calibration = None

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    # ---- engine thread ------------------------------------------------------------------

    def _after_config_change(self, changed: dict[str, Any]) -> None:
        for name in changed:
            restart = PARAMS_BY_NAME[name].restart
            if restart:
                self._restart.add(restart)
        self.smoother.alpha = self.ref_smoother.alpha = self.cfg.smoothing_alpha
        self.stabilizer.lookback = self.cfg.click_lookback
        self.stabilizer.drag_deadzone = self.cfg.drag_deadzone
        self.stabilizer.release_hold = self.cfg.release_hold

    def _take_restarts(self) -> set[str]:
        with self._lock:
            restart, self._restart = self._restart, set()
        return restart

    def _run(self) -> None:
        cam: Camera | None = None
        tracker: HandTracker | None = None
        log = SessionLog() if self.cfg.session_log else None
        try:
            while not self._stop.is_set():
                restart = self._take_restarts()
                if "camera" in restart and cam is not None:
                    cam.release()
                    cam = None
                if "tracker" in restart and tracker is not None:
                    tracker.close()
                    tracker = None

                if cam is None:
                    cam = self._open_camera()
                    if cam is None:
                        continue
                if tracker is None:
                    tracker = HandTracker(self.cfg)

                frame = cam.read()
                if frame is None:
                    cam.release()
                    cam = None
                    self._stop_control()  # never leave a dragged button down during an outage
                    self._publish_error("Camera stopped sending frames; retrying…")
                    self._stop.wait(CAMERA_RETRY_S)
                    continue
                self._step(frame, tracker, cam.size, log)
        finally:
            self._stop_control()
            if cam is not None:
                cam.release()
            if tracker is not None:
                tracker.close()
            if log is not None:
                log.close()

    def _open_camera(self) -> Camera | None:
        try:
            return Camera(self.cfg.camera_index, self.cfg.frame_width, self.cfg.frame_height)
        except RuntimeError as exc:
            self._publish_error(f"{exc}; retrying…")
            self._stop.wait(CAMERA_RETRY_S)
            return None

    def _stop_control(self) -> None:
        self.cursor.release()
        self.gestures.reset()
        self.smoother.reset()
        self.ref_smoother.reset()
        self.stabilizer.reset()

    def _step(self, frame: np.ndarray, tracker: HandTracker, frame_size: tuple[int, int],
              log: SessionLog | None) -> None:
        hands = tracker.process(frame)
        hand = hands[0] if hands else None
        now = time.perf_counter()

        calibrating = self._update_calibration(hand, frame_size, now)
        wanted = self._enabled and not calibrating
        if self._control_active and not wanted:
            self._stop_control()
        self._control_active = wanted

        gesture: GestureOutput | None = None
        if self._control_active:
            try:
                gesture = self.gestures.update(hand, frame_size, now)
                self._drive_mouse(hand, gesture, frame_size, now)
            except pyautogui.FailSafeException:
                # Mouse slammed into a corner: pause control rather than quitting the app.
                self._enabled = self._control_active = False
                self._stop_control()
                self._notice = "Failsafe: mouse hit a screen corner, hand control paused"
            if log is not None:
                log.write(now, hand, gesture)

        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_frame_t, 1e-6)
        self._last_frame_t = now
        state = self._build_state(hand, gesture, frame_size, now)
        with self._lock:
            self._snapshot = Snapshot(frame, hand, state)

        if self.on_update is not None:
            self.on_update(state, self._encode(frame) if self.want_jpeg else None)

    def _update_calibration(self, hand: Hand | None, frame_size: tuple[int, int],
                            now: float) -> bool:
        """Collect aim points while calibrating; True while a calibration is running."""
        with self._lock:
            cal = self._calibration
        if cal is None or not cal.active:
            return False
        if cal.started is None:
            cal.started = now
        if hand is not None:
            cal.points.append(pointer_position(hand, frame_size, self.cfg.fingertip_follow))
        if now - cal.started >= CALIBRATION_S:
            try:
                cal.result = fit_area(np.array(cal.points))
            except ValueError as exc:
                cal.error = str(exc)
            self._area_offset[:] = 0.0
            return False
        return True

    def _zone_state(self, now: float) -> dict[str, float]:
        """Which edges of the hand area are taskbar zones, as fractions of the area."""
        zone = self.cfg.taskbar_zone
        return {edge: (zone if band > 0 else 0.0)
                for edge, band in zip(("left", "top", "right", "bottom"), self.cursor.bands(now))}

    def _calibration_state(self, now: float) -> dict[str, Any] | None:
        cal = self._calibration
        if cal is None:
            return None
        elapsed = 0.0 if cal.started is None else now - cal.started
        return {"id": cal.id, "active": cal.active,
                "remaining": max(0.0, round(CALIBRATION_S - elapsed, 1)),
                "seen": len(cal.points), "result": cal.result, "error": cal.error}

    def _drive_mouse(self, hand: Hand | None, gesture: GestureOutput,
                     frame_size: tuple[int, int], now: float) -> None:
        if hand is None:
            if self._hand_lost_since is None:
                self._hand_lost_since = now
            elif now - self._hand_lost_since >= AREA_HOME_AFTER_S:
                self._area_offset[:] = 0.0
        else:
            self._hand_lost_since = None

        if gesture.mode in (Mode.NO_HAND, Mode.SCROLL):
            # Hold the cursor still; restart smoothing so it doesn't glide later.
            self.smoother.reset()
            self.ref_smoother.reset()
            self.stabilizer.reset()
        elif hand is None or gesture.releasing:
            # Brief tracking dropout during a press, or a grab whose hand looks open while
            # the drop is confirmed: hold still, keeping the aim pin, so the cursor neither
            # jumps with the button down nor drifts off where the item is being dropped.
            pass
        else:
            aim = pointer_position(hand, frame_size, self.cfg.fingertip_follow)
            self._slide_area(aim, frame_size)
            region = self._region(frame_size)
            bands = self.cursor.bands(now)

            def to_screen(norm_xy: np.ndarray) -> np.ndarray:
                return map_to_screen(norm_xy, frame_size, region, self.cursor.screen_size,
                                     bands, self.cfg.taskbar_zone)

            pointer = self.stabilizer.update(
                self.smoother.update(to_screen(aim)),
                self.ref_smoother.update(to_screen(hand.landmarks[INDEX_MCP, :2])), now,
                pressed=gesture.press_started,
                held=gesture.mode in (Mode.LEFT_PRESS, Mode.RIGHT_PRESS, Mode.GRAB,
                                      Mode.TOUCH, Mode.PINCH),
                draggable=False,  # taps never drag; the fist does
                grab=gesture.mode is Mode.GRAB,
            )
            self.cursor.move(pointer)
            self._pointer = pointer

        for event in gesture.events:
            if event is Event.LEFT_CLICK:
                self._left_click(now)
            elif event is Event.DOUBLE_CLICK:
                self._last_click = None             # don't pair it with an earlier click
                self.cursor.double_click()
                print("Double click (pinch)")
            elif event is Event.LEFT_DOWN:
                self.cursor.left_down()
            elif event is Event.LEFT_UP:
                self.cursor.left_up()
            elif event is Event.RIGHT_CLICK:
                self.cursor.right_click()
        self.cursor.scroll(gesture.scroll)

    def _home_region(self, frame_size: tuple[int, int]) -> Region:
        return active_region(frame_size, self.cursor.screen_size, self.cfg.cursor_speed,
                             self.cfg.area_center_y, self.cfg.area_height,
                             self.cfg.area_center_x)

    def _region(self, frame_size: tuple[int, int]) -> Region:
        x0, y0, x1, y1 = self._home_region(frame_size)
        dx, dy = self._area_offset
        return x0 + dx, y0 + dy, x1 + dx, y1 + dy

    def _slide_area(self, aim: np.ndarray, frame_size: tuple[int, int]) -> None:
        """Sliding control area: same size (so the same cursor speed), but when the hand
        goes past an edge the area moves with it instead of the hand hitting a wall. The
        cursor stays pinned to that screen edge, and moving back moves it straight away.
        The area stays inside the camera frame."""
        fw, fh = frame_size
        ax, ay = aim[0] * fw, aim[1] * fh
        x0, y0, x1, y1 = self._region(frame_size)
        self._area_offset[0] += min(0.0, ax - x0) + max(0.0, ax - x1)
        self._area_offset[1] += min(0.0, ay - y0) + max(0.0, ay - y1)
        hx0, hy0, hx1, hy1 = self._home_region(frame_size)
        self._area_offset[0] = min(max(self._area_offset[0], -hx0), fw - hx1)
        self._area_offset[1] = min(max(self._area_offset[1], -hy0), fh - hy1)

    def _left_click(self, now: float) -> None:
        """Click, turning a second click on the same spot within `double_click_window`
        into a real double click. Windows wants both within ~0.5 s and on the same pixel,
        which is hard to hit with gestures (real double touches came 0.5-0.9 s apart, and
        a touch only clicks as the fingers open). The second click returns to the first
        one's exact spot; if it came too late for Windows to pair them, a double click is
        sent outright. Clicks on different spots stay separate single clicks."""
        last = self._last_click
        here = self._pointer
        radius = DOUBLE_CLICK_RADIUS * self.cursor.screen_size[0]
        if (last is not None and here is not None
                and now - last[0] <= self.cfg.double_click_window
                and float(np.linalg.norm(here - last[1])) <= radius):
            self._last_click = None                     # never chain into a triple click
            self.cursor.move(last[1])
            if now - last[0] <= self._double_click_time * 0.9:
                self.cursor.left_click()                # Windows pairs it with the first
            else:
                self.cursor.double_click()
            print("Double click")
            return
        self.cursor.left_click()
        print("Click")
        if here is not None:
            self._last_click = (now, here.copy())

    def _build_state(self, hand: Hand | None, gesture: GestureOutput | None,
                     frame_size: tuple[int, int], now: float) -> dict[str, Any]:
        fw, fh = frame_size
        x0, y0, x1, y1 = self._region(frame_size)
        notice, self._notice = self._notice, None
        return {
            "type": "state",
            "fps": round(self._fps, 1),
            "enabled": self._enabled,
            "error": None,
            "notice": notice,
            "frame_size": [fw, fh],
            "active_region": [x0 / fw, y0 / fh, x1 / fw, y1 / fh],
            "calibration": self._calibration_state(now),
            "taskbar_zone": self._zone_state(now),
            "hand": None if hand is None else {
                "handedness": hand.handedness,
                "score": round(hand.score, 3),
                "landmarks": np.round(hand.landmarks, 4).tolist(),
            },
            "mode": (gesture.mode if gesture else Mode.NO_HAND if hand is None
                     else Mode.POINT).name,
            "events": [e.name for e in gesture.events] if gesture else [],
            "scroll": gesture.scroll if gesture else 0,
            "scroll_anchor": (list(gesture.scroll_anchor)
                              if gesture and gesture.scroll_anchor else None),
            "left_push": round(gesture.left_push, 3) if gesture else 0.0,
            "right_push": round(gesture.right_push, 3) if gesture else 0.0,
        }

    def _publish_error(self, message: str) -> None:
        state = {"type": "state", "fps": 0.0, "enabled": self._enabled, "error": message,
                 "notice": None, "hand": None, "mode": Mode.NO_HAND.name, "events": [],
                 "scroll": 0, "left_push": 0.0, "right_push": 0.0}
        with self._lock:
            self._snapshot = Snapshot(None, None, state)
        if self.on_update is not None:
            self.on_update(state, None)

    @staticmethod
    def _encode(frame: np.ndarray) -> bytes | None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None
