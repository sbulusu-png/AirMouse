"""Engine behaviour with a fake camera, tracker and mouse: no hardware, no real cursor."""

import time
from collections.abc import Sequence

import numpy as np
import pytest

import airmouse.engine as engine_mod
from airmouse.config import Config
from airmouse.engine import Engine
from airmouse.tracker import Hand

from .hands import FS, REST, make_hand, tap

FRAME = np.zeros((FS[1], FS[0], 3), np.uint8)


class FakeCursor:
    screen_size = (1920, 1200)

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.held = False
        self.taskbar = (0, 0, 0, 0)   # no taskbar unless a test sets one

    def bands(self, now):
        return self.taskbar

    def move(self, xy):
        self.calls.append(("move", float(xy[0]), float(xy[1])))

    def left_down(self):
        self.held = True
        self.calls.append(("left_down",))

    def left_up(self):
        self.held = False
        self.calls.append(("left_up",))

    def left_click(self):
        self.calls.append(("left_click",))

    def double_click(self):
        self.calls.append(("double_click",))

    def right_click(self):
        self.calls.append(("right_click",))

    def scroll(self, amount):
        if amount:
            self.calls.append(("scroll", amount))

    def release(self):
        if self.held:
            self.left_up()

    def moves(self) -> list[tuple[float, float]]:
        return [(c[1], c[2]) for c in self.calls if c[0] == "move"]


class ScriptedTracker:
    def __init__(self, script: Sequence[Hand | None]) -> None:
        self.script = list(script)

    def process(self, frame):
        h = self.script.pop(0) if self.script else None
        return [] if h is None else [h]

    def close(self):
        pass


class FakeClock:
    """Stands in for the engine's `time` module: each reading advances one camera frame."""

    def __init__(self) -> None:
        self.t = 0.0

    def perf_counter(self) -> float:
        self.t += 1 / 30
        return self.t


@pytest.fixture
def engine(monkeypatch) -> Engine:
    monkeypatch.setattr(engine_mod, "time", FakeClock())
    e = Engine(Config(), enabled=True)
    e.cursor = FakeCursor()  # type: ignore[assignment]
    return e


def step_all(e: Engine, script: Sequence[Hand | None]) -> None:
    tracker = ScriptedTracker(script)
    for _ in range(len(script)):
        e._step(FRAME, tracker, FS, None)  # type: ignore[arg-type]


FIST = make_hand(ext=())


def test_tap_clicks_through_engine(engine):
    step_all(engine, [REST] * 10 + tap("i") + [REST] * 3)
    kinds = [c[0] for c in engine.cursor.calls if c[0] != "move"]
    assert kinds == ["left_click"]


def test_pushing_deeper_never_moves_cursor(engine):
    step_all(engine, [REST] * 10)
    before = engine.cursor.moves()[-1]
    n_before = len(engine.cursor.calls)
    step_all(engine, tap("i", frames_down=10, amount=0.9) + [REST] * 2)
    calls = engine.cursor.calls[n_before:]
    click = next(i for i, c in enumerate(calls) if c[0] == "left_click")
    # From the moment the push is detected until the click, the cursor is pinned.
    pinned = [c[1:] for c in calls[2:click] if c[0] == "move"]
    assert len(set(pinned)) == 1
    x, y = pinned[0]
    assert abs(x - before[0]) < 15 and abs(y - before[1]) < 15  # and where we aimed
    assert calls[click - 1][1:] == pinned[0]                    # click lands there


def test_moving_hand_during_push_unfreezes_cursor_without_click(engine):
    step_all(engine, [REST] * 10)
    start = engine.cursor.moves()[-1]
    moving = []
    for k, h in enumerate(tap("i", frames_down=10)):
        lm = h.landmarks.copy()
        lm[:, 0] += 0.012 * k                    # hand sweeps right while "pushing"
        moving.append(Hand(lm, "Right", 0.99))
    after = Hand(moving[-1].landmarks, "Right", 0.99)
    step_all(engine, moving + [after] * 8)      # pinned ~0.15 s after the cancel, then free
    assert "left_click" not in [c[0] for c in engine.cursor.calls]
    assert engine.cursor.moves()[-1][0] - start[0] > 150  # cursor followed the hand


def test_fist_carries_cursor_and_drops_where_opened(engine):
    fist = make_hand(ext=())
    step_all(engine, [REST] * 10)
    aim = engine.cursor.moves()[-1]
    step_all(engine, [fist] * 6)
    cur = engine.cursor
    assert cur.held
    assert abs(cur.moves()[-1][0] - aim[0]) < 15 and abs(cur.moves()[-1][1] - aim[1]) < 40

    carried = []
    for k in range(1, 16):
        lm = fist.landmarks.copy()
        lm[:, 0] += 0.006 * k                    # fist slides right
        carried.append(Hand(lm, "Right", 0.99))
    step_all(engine, carried)
    drop_at = cur.moves()[-1]
    assert drop_at[0] - aim[0] > 100             # the item travelled with the fist

    opened = make_hand(dx=0.006 * 15 * FS[0])
    step_all(engine, [opened] * 6)
    assert not cur.held                          # hand opened -> dropped
    up = max(i for i, c in enumerate(cur.calls) if c[0] == "left_up")
    last_move_before_up = [c for c in cur.calls[:up] if c[0] == "move"][-1][1:]
    assert last_move_before_up == drop_at        # released exactly where it was carried


def test_dropout_during_grab_keeps_cursor_pinned(engine):
    step_all(engine, [REST] * 10 + [FIST] * 6)
    cur = engine.cursor
    assert cur.held
    pinned = cur.moves()[-1]
    step_all(engine, [None, FIST, FIST])
    assert cur.held                             # brief dropout: still holding
    assert cur.moves()[-1] == pinned            # and the cursor didn't jump


def test_disabling_control_releases_held_button(engine):
    step_all(engine, [REST] * 10 + [FIST] * 6)
    assert engine.cursor.held
    engine.set_enabled(False)
    step_all(engine, [FIST])
    assert not engine.cursor.held


def test_camera_failure_mid_drag_releases_button(monkeypatch):
    frames_left = {"n": 25}
    opened = {"n": 0}

    class FakeCamera:
        size = FS

        def __init__(self, *args, **kwargs):
            opened["n"] += 1
            if opened["n"] > 1:
                raise RuntimeError("camera unplugged")

        def read(self):
            time.sleep(1 / 30)  # real frame pacing: the fist must be held FIST_ENTER_S
            frames_left["n"] -= 1
            return FRAME if frames_left["n"] >= 0 else None

        def release(self):
            pass

    script = [REST] * 10 + [FIST] * 20  # still grabbing when the camera dies
    monkeypatch.setattr(engine_mod, "Camera", FakeCamera)
    monkeypatch.setattr(engine_mod, "HandTracker", lambda cfg: ScriptedTracker(script))
    monkeypatch.setattr(engine_mod, "CAMERA_RETRY_S", 0.05)

    e = Engine(Config(), enabled=True)
    cursor = FakeCursor()
    e.cursor = cursor  # type: ignore[assignment]
    e.start()
    try:
        deadline = time.time() + 5
        while opened["n"] < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert opened["n"] >= 2, "engine never noticed the dead camera"
        assert e.alive                                   # still retrying, not crashed
        kinds = [c[0] for c in cursor.calls if c[0] != "move"]
        assert kinds == ["left_down", "left_up"]         # released during the outage
        assert e.snapshot().state["error"]
    finally:
        e.stop()


# --- Double clicks ------------------------------------------------------------------------

from .hands import touch  # noqa: E402


def clicks(cur: FakeCursor) -> list[str]:
    return [c[0] for c in cur.calls if c[0] in ("left_click", "double_click")]


def test_quick_second_touch_is_paired_by_windows(engine):
    step_all(engine, [REST] * 10 + touch() + [REST] * 2 + touch() + [REST] * 3)
    assert clicks(engine.cursor) == ["left_click", "left_click"]   # < 0.45 s apart


def test_slower_second_touch_becomes_a_real_double_click(engine):
    step_all(engine, [REST] * 10 + touch() + [REST] * 10 + touch() + [REST] * 3)
    assert clicks(engine.cursor) == ["left_click", "double_click"]  # ~0.7 s apart


def test_second_click_returns_to_first_clicks_spot(engine):
    step_all(engine, [REST] * 10 + touch())
    first = [c for c in engine.cursor.calls if c[0] == "move"][-1][1:]
    drift = [make_hand(dx=2.0 * k) for k in range(1, 8)]          # hand drifts a little
    step_all(engine, drift + touch() + [REST] * 3)
    assert clicks(engine.cursor) == ["left_click", "double_click"]
    i = max(i for i, c in enumerate(engine.cursor.calls) if c[0] in ("left_click", "double_click"))
    assert engine.cursor.calls[i - 1][1:] == first


def test_clicks_on_different_spots_stay_single(engine):
    """Two quick clicks on different items must not merge into a double click."""
    step_all(engine, [REST] * 10 + touch())
    move_away = [make_hand(dx=12.0 * k) for k in range(1, 8)]     # ~250 px on screen
    step_all(engine, move_away + touch() + [REST] * 3)
    assert clicks(engine.cursor) == ["left_click", "left_click"]


def test_slow_natural_double_touch_still_double_clicks(engine):
    """User report: double touches acted as two single clicks; natural pairs can be
    slower than the recorded 0.5-0.9 s."""
    step_all(engine, [REST] * 10 + touch() + [REST] * 30 + touch() + [REST] * 3)  # ~1.3 s
    assert clicks(engine.cursor) == ["left_click", "double_click"]


def test_pairing_never_chains_into_a_triple_and_far_clicks_stay_single(engine):
    # click, pair (double), then a third right after: it starts a new pair, not a triple
    step_all(engine, [REST] * 10 + touch() + [REST] * 10 + touch() + [REST] * 10 + touch()
             + [REST] * 40 + touch() + [REST] * 3)
    assert clicks(engine.cursor) == ["left_click", "double_click", "left_click", "left_click"]


# --- Cursor anchor: fingers gesturing must not move the cursor ------------------------------

from airmouse.engine import pointer_position  # noqa: E402

from .hands import make_touch  # noqa: E402


def test_closing_fingertips_for_a_touch_does_not_move_cursor(engine):
    """User report: closing the fingertips moved the cursor before the click."""
    step_all(engine, [REST] * 15)
    before = engine.cursor.moves()[-1]
    closing = [make_touch(k / 8) for k in range(1, 9)]     # slow close, 8 frames
    step_all(engine, closing)
    after = engine.cursor.moves()[-1]
    assert abs(after[0] - before[0]) < 3 and abs(after[1] - before[1]) < 3


def test_curling_into_fist_does_not_move_cursor(engine):
    step_all(engine, [REST] * 15)
    before = engine.cursor.moves()[-1]
    curl = [make_hand(push="imrp", amount=0.25 * k) for k in range(1, 4)] + [make_hand(ext=())]
    step_all(engine, curl)
    after = engine.cursor.moves()[-1]
    assert abs(after[0] - before[0]) < 3 and abs(after[1] - before[1]) < 3


def test_pointer_aims_where_the_straight_finger_points():
    aim = pointer_position(REST, FS, 0.0) * FS
    tip = REST.landmarks[8, :2] * FS
    assert abs(aim[0] - tip[0]) < 45 and abs(aim[1] - tip[1]) < 10   # same height as tip
    full = pointer_position(REST, FS, 1.0) * FS
    assert abs(full[0] - tip[0]) < 1e-3 and abs(full[1] - tip[1]) < 1e-3  # follow = 1: tip


# --- Sliding control area ---------------------------------------------------------------------

def _hand_at(dx_px: float) -> Hand:
    return make_hand(dx=dx_px)


def test_area_slides_with_hand_and_cursor_returns_immediately(engine):
    """User request: a bigger area to move in, at the same cursor speed. Past the edge the
    area slides along; moving back must move the cursor at once, not after re-entering."""
    step_all(engine, [_hand_at(0)] * 10)
    right = [_hand_at(20.0 * k) for k in range(1, 30)]           # far past the area's edge
    step_all(engine, right + [right[-1]] * 15)
    at_edge = engine.cursor.moves()[-1][0]
    assert at_edge >= engine.cursor.screen_size[0] - 3           # pinned to the right edge
    back = [_hand_at(20.0 * 29 - 8.0 * k) for k in range(1, 8)]   # a little to the left
    step_all(engine, back + [back[-1]] * 15)
    assert engine.cursor.moves()[-1][0] < at_edge - 100          # followed straight away


def test_slid_area_keeps_cursor_speed(engine):
    """The area slides, it doesn't grow: the same hand motion moves the cursor as far."""
    step_all(engine, [_hand_at(0)] * 20)
    a = engine.cursor.moves()[-1][0]
    step_all(engine, [_hand_at(30.0)] * 20)
    gain_home = engine.cursor.moves()[-1][0] - a
    # Slide the area right (hand stays inside the camera frame), then measure there.
    step_all(engine, [_hand_at(20.0 * k) for k in range(2, 21)] + [_hand_at(400.0)] * 10)
    assert engine._area_offset[0] > 0
    step_all(engine, [_hand_at(360.0)] * 20)
    b = engine.cursor.moves()[-1][0]
    step_all(engine, [_hand_at(330.0)] * 20)
    gain_slid = b - engine.cursor.moves()[-1][0]
    assert abs(gain_home - gain_slid) < 0.05 * gain_home


def test_area_goes_home_after_hand_leaves(engine):
    step_all(engine, [_hand_at(20.0 * k) for k in range(1, 30)])
    assert engine._area_offset[0] > 0
    step_all(engine, [None] * 60)                                  # 2 s without a hand
    assert engine._area_offset[0] == 0


# --- "Fit area to my reach" calibration ---------------------------------------------------------

from airmouse.engine import CALIBRATION_S, fit_area  # noqa: E402


def test_fit_area_matches_the_reach():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.30, 0.70, 400), rng.uniform(0.15, 0.55, 400)])
    fit = fit_area(pts)
    width = (0.70 - 0.30) * 0.94 * 0.9                      # p3-p97 of the sweep, inset
    assert abs(fit["cursor_speed"] - 1 / width) < 0.15
    assert abs(fit["area_height"] - 0.40 * 0.94 * 0.9) < 0.03
    assert abs(fit["area_center_x"] - 0.50) < 0.02
    assert abs(fit["area_center_y"] - 0.35) < 0.02


def test_fit_area_follows_an_off_centre_reach():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0.45, 0.85, 300), rng.uniform(0.20, 0.50, 300)])
    assert fit_area(pts)["area_center_x"] > 0.6


@pytest.mark.parametrize("pts, advice", [
    (np.full((20, 2), 0.5), "long enough"),                   # hand barely seen
    (np.full((200, 2), 0.5) + 0.01, "barely moved"),          # hand seen but still
])
def test_fit_area_explains_unusable_sweeps(pts, advice):
    with pytest.raises(ValueError, match=advice):
        fit_area(pts)


def test_calibration_pauses_the_mouse_then_proposes_settings(engine):
    step_all(engine, [REST] * 5)
    engine.cursor.calls.clear()
    engine.start_calibration()
    frames = int(CALIBRATION_S * 30) + 5
    sweep = [make_hand(dx=200 * np.sin(k / 9), dy=-150 + 120 * np.cos(k / 13)) for k in range(frames)]
    sweep[40:44] = touch()[:4]                                 # a gesture mid-sweep: no click
    step_all(engine, sweep[:-10])
    assert engine.snapshot().state["calibration"]["active"] is True
    assert engine.cursor.calls == []                          # mouse untouched during the sweep
    step_all(engine, sweep[-10:])
    cal = engine.snapshot().state["calibration"]
    assert cal["active"] is False and cal["error"] is None
    assert set(cal["result"]) == {"cursor_speed", "area_height", "area_center_x", "area_center_y"}
    step_all(engine, [REST] * 5)
    assert engine.cursor.moves()                              # control resumes afterwards


def test_calibration_can_be_cancelled(engine):
    engine.start_calibration()
    step_all(engine, [REST] * 10)
    assert engine.snapshot().state["calibration"]["active"] is True
    engine.cancel_calibration()
    step_all(engine, [REST] * 5)
    assert engine.snapshot().state["calibration"] is None
    assert engine.cursor.moves()


# --- Taskbar zone --------------------------------------------------------------------------------

def test_taskbar_is_reached_higher_and_has_room_to_aim(engine):
    """User report: taskbar apps are hard to click, way down at the screen's bottom edge."""
    engine.cursor.taskbar = (0, 0, 0, 60)          # 60 px taskbar at the bottom of 1200
    sh = engine.cursor.screen_size[1]
    ys = {}
    for dy in range(0, 260, 10):                   # lower the hand step by step
        step_all(engine, [make_hand(dy=float(dy))] * 25)
        ys[dy] = engine.cursor.moves()[-1][1]
    on_bar = [dy for dy, y in ys.items() if y >= sh - 60]
    assert on_bar, "never reached the taskbar"
    # Several hand heights land on the 60 px strip: room to wobble without slipping off.
    assert len(on_bar) >= 3


def test_no_taskbar_means_plain_linear_mapping():
    from airmouse.cursor import map_to_screen, NO_BANDS
    region = (100.0, 100.0, 500.0, 400.0)
    for v in (0.1, 0.3, 0.5):
        a = map_to_screen(np.array([v, v]), (1000, 1000), region, (1920, 1200), NO_BANDS, 0.15)
        b = map_to_screen(np.array([v, v]), (1000, 1000), region, (1920, 1200))
        assert np.allclose(a, b)


def test_pinch_sends_one_double_click(engine):
    from .hands import pinch
    step_all(engine, [REST] * 10 + pinch() + [REST] * 5)
    assert [c[0] for c in engine.cursor.calls if c[0] != "move"] == ["double_click"]


# --- Hand control off = camera off --------------------------------------------------------------

class CountingCamera:
    """Fake camera that records opens and releases (the light being on)."""
    opened = 0
    released = 0
    size = FS

    def __init__(self, *args, **kwargs):
        type(self).opened += 1

    def read(self):
        time.sleep(1 / 30)
        return FRAME

    def release(self):
        type(self).released += 1

    @classmethod
    def on(cls) -> bool:
        return cls.opened > cls.released


def _wait_for(cond, timeout=5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def counting_engine(monkeypatch):
    CountingCamera.opened = CountingCamera.released = 0
    monkeypatch.setattr(engine_mod, "Camera", CountingCamera)
    monkeypatch.setattr(engine_mod, "HandTracker", lambda cfg: ScriptedTracker([]))
    monkeypatch.setattr(engine_mod, "IDLE_CHECK_S", 0.05)
    engines = []

    def make(enabled: bool) -> Engine:
        e = Engine(Config(), enabled=enabled)
        e.cursor = FakeCursor()  # type: ignore[assignment]
        e.start()
        engines.append(e)
        return e

    yield make
    for e in engines:
        e.stop()


def test_turning_hand_control_off_turns_the_camera_off(counting_engine):
    """User report: the camera stayed on after switching hand control off."""
    e = counting_engine(True)
    assert _wait_for(CountingCamera.on)
    e.set_enabled(False)
    assert _wait_for(lambda: not CountingCamera.on())
    assert e.snapshot().state["camera"] is False
    e.set_enabled(True)
    assert _wait_for(CountingCamera.on)                       # back on with hand control
    assert e.snapshot().state.get("camera") is True or _wait_for(
        lambda: e.snapshot().state.get("camera") is True)


def test_starting_paused_never_opens_the_camera(counting_engine):
    e = counting_engine(False)
    time.sleep(0.4)
    assert CountingCamera.opened == 0
    assert e.snapshot().state["camera"] is False


def test_calibrating_while_paused_uses_the_camera_only_for_the_sweep(counting_engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "CALIBRATION_S", 0.3)
    e = counting_engine(False)
    e.start_calibration()
    assert _wait_for(CountingCamera.on)
    assert _wait_for(lambda: not CountingCamera.on())         # off again once it's done
    assert e.snapshot().state["calibration"]["active"] is False
