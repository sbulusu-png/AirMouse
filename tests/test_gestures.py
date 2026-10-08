from airmouse.config import Config
from airmouse.gestures import Event, GestureEngine, Mode, measure

from .hands import FS, REST, events, fold, make_hand, run, tap


def engine() -> GestureEngine:
    return GestureEngine(Config())


def test_rest_pose_is_point_with_no_events():
    outs, _ = run(engine(), [REST] * 30)
    assert all(o.mode is Mode.POINT for o in outs)
    assert events(outs) == []


def test_index_tap_is_one_left_click():
    outs, _ = run(engine(), [REST] * 10 + tap("i") + [REST] * 5)
    assert events(outs) == [Event.LEFT_CLICK]


def test_click_fires_on_pull_back_not_on_push():
    outs, _ = run(engine(), [REST] * 10 + tap("i", frames_down=8) + [REST] * 5)
    pressed = next(i for i, o in enumerate(outs) if o.press_started)
    clicked = next(i for i, o in enumerate(outs) if Event.LEFT_CLICK in o.events)
    assert pressed < 13 < 13 + 8 < clicked   # aim pinned at the push, click on the way back


def test_double_tap_fits_windows_double_click_time():
    outs, _ = run(engine(), [REST] * 10 + tap("i") + [REST] * 2 + tap("i") + [REST] * 3)
    assert events(outs) == [Event.LEFT_CLICK] * 2
    clicks = [i for i, o in enumerate(outs) if Event.LEFT_CLICK in o.events]
    assert (clicks[1] - clicks[0]) / 30 < 0.5  # Windows' default double-click time


def test_thumb_fold_is_one_right_click():
    outs, _ = run(engine(), [REST] * 10 + fold() + [REST] * 5)
    assert events(outs) == [Event.RIGHT_CLICK]


def test_middle_finger_tap_no_longer_clicks():
    outs, _ = run(engine(), [REST] * 10 + tap("m") + [REST] * 5)
    assert events(outs) == []


def test_thumb_already_tucked_is_not_a_right_click():
    """Real data: the thumb often rests tucked in while pointing or scrolling."""
    tucked = make_hand(ext=("i",), thumb_fold=1.0)
    outs, _ = run(engine(), [tucked] * 40)
    assert events(outs) == []


def test_slow_thumb_drift_is_not_a_right_click():
    slow = [make_hand(thumb_fold=k / 40) for k in range(41)]
    outs, _ = run(engine(), [REST] * 10 + slow + slow[::-1] + [REST] * 5)
    assert events(outs) == []


def test_thumb_fold_with_fingers_curling_is_not_a_right_click():
    """Switching to pointing tucks the thumb AND curls three fingers: not a fold."""
    to_point = [make_hand(ext=("i",), thumb_fold=1.0)] * 8
    outs, _ = run(engine(), [REST] * 10 + to_point + [REST] * 8)
    assert Event.RIGHT_CLICK not in events(outs)


def test_held_thumb_fold_is_a_pose_not_a_click():
    outs, _ = run(engine(), [REST] * 10 + fold(frames_in=45) + [REST] * 5)
    assert events(outs) == []


def test_long_push_is_a_posture_never_a_click_or_held_button():
    """Real data: pointing at the camera foreshortens the index exactly like a push; it
    used to hold the left button down for seconds."""
    outs, _ = run(engine(), [REST] * 10 + tap("i", frames_down=45) + [REST] * 5)
    assert events(outs) == []
    assert not any(o.mode is Mode.GRAB for o in outs)


def test_held_push_is_adopted_as_rest_then_taps_work_from_there():
    e = engine()
    pointing = make_hand(push="i", amount=0.4)
    outs, t = run(e, [REST] * 10 + tap("i", amount=0.4, frames_down=45)[:-3] + [pointing] * 10)
    assert events(outs) == []
    deeper = [make_hand(push="i", amount=0.4 + 0.4 * k / 3) for k in range(1, 4)]
    outs, _ = run(e, deeper + [deeper[-1]] * 3 + deeper[::-1] + [pointing] * 5, t)
    assert events(outs) == [Event.LEFT_CLICK]


def test_slow_drift_over_threshold_is_not_a_click():
    """Real data: the index crept to just under the threshold, then noise tipped it over."""
    slow = [make_hand(push="i", amount=0.6 * k / 45) for k in range(45)]
    outs, _ = run(engine(), [REST] * 10 + slow + slow[::-1] + [REST] * 5)
    assert events(outs) == []


def test_tap_while_moving_hand_is_cancelled():
    moving = []
    for k, h in enumerate(tap("i", frames_down=6)):
        lm = h.landmarks.copy()
        lm[:, 0] += 0.012 * k          # hand sweeps sideways during the "tap"
        moving.append(type(h)(lm, "Right", 0.99))
    outs, _ = run(engine(), [REST] * 10 + moving + [REST] * 5)
    assert events(outs) == []


def test_small_wiggle_does_not_click():
    outs, _ = run(engine(), [REST] * 10 + tap("i", amount=0.15) + [REST] * 5)
    assert events(outs) == []


def test_tilting_whole_hand_does_not_click():
    tilting = [make_hand(tilt=1 - 0.1 * k) for k in range(5)] + [make_hand(tilt=0.6)] * 5
    outs, _ = run(engine(), [REST] * 10 + tilting)
    assert events(outs) == []


def test_slow_posture_change_is_absorbed_then_tap_still_works():
    e = engine()
    slow = [make_hand(push="i", amount=0.3 * k / 60) for k in range(60)]
    settled = make_hand(push="i", amount=0.3)
    outs, t = run(e, [REST] * 5 + slow + [settled] * 30)
    assert events(outs) == []
    deeper = [make_hand(push="i", amount=0.3 + 0.5 * k / 3) for k in range(1, 4)]
    outs, _ = run(e, deeper + [make_hand(push="i", amount=0.8)] * 3 + deeper[::-1]
                  + [settled] * 5, t)
    assert events(outs) == [Event.LEFT_CLICK]


def test_cooldown_suppresses_rapid_second_press_without_late_fire():
    cfg = Config()
    cfg.click_cooldown = 1.0
    e = GestureEngine(cfg)
    outs, t = run(e, [REST] * 10 + tap("i") + [REST] * 2)
    assert events(outs) == [Event.LEFT_CLICK]
    # Second push inside the cooldown: ignored, and must not fire when cooldown expires.
    outs, t = run(e, tap("i", frames_down=40) + [REST] * 5, t)
    assert events(outs) == []


def test_hand_lost_mid_tap_does_not_click():
    e = engine()
    outs, t = run(e, [REST] * 10 + tap("i", frames_down=10)[:-3] + [None] * 15)
    assert events(outs) == []


def test_hand_lost_while_idle_reports_no_hand():
    outs, _ = run(engine(), [REST] * 5 + [None])
    assert outs[-1].mode is Mode.NO_HAND and outs[-1].events == []


def test_scroll_pose_detected_only_for_two_fingers_together():
    assert measure(make_hand(ext=("i", "m")), FS).scroll_pose
    assert not measure(REST, FS).scroll_pose                     # open palm
    assert not measure(make_hand(ext=("i",)), FS).scroll_pose     # pointing


def test_scroll_direction_follows_hand():
    up, _ = run(engine(), [make_hand(ext=("i", "m"), dy=-3 * k) for k in range(30)])
    down, _ = run(engine(), [make_hand(ext=("i", "m"), dy=3 * k) for k in range(30)])
    assert up[-1].mode is Mode.SCROLL
    assert sum(o.scroll for o in up) > 0 > sum(o.scroll for o in down)
    assert events(up) == events(down) == []


def test_still_scroll_pose_does_not_scroll():
    outs, _ = run(engine(), [make_hand(ext=("i", "m"))] * 30)
    assert outs[-1].mode is Mode.SCROLL
    assert sum(o.scroll for o in outs) == 0


FIST = make_hand(ext=())
POINTING = make_hand(ext=("i",))


def test_fist_grabs_and_open_hand_drops():
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 20 + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]
    assert sum(o.mode is Mode.GRAB for o in outs) >= 15
    assert sum(o.press_started for o in outs) == 1


def test_motion_blur_glitch_during_fast_drag_keeps_grip():
    """User report: moving the fist faster lost the grip. Blur can make the fingers look
    open for a frame or two; that isn't letting go."""
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 6
                  + ([REST] * 3 + [FIST] * 4) * 5 + [FIST] * 3 + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]
    assert any(o.releasing for o in outs)        # the glitch frames hold the cursor still


def test_tracking_dropout_during_fast_drag_keeps_grip():
    e = engine()
    outs, t = run(e, [REST] * 10 + [FIST] * 6 + [None] * 15 + [FIST] * 5)  # 0.5 s lost
    assert events(outs) == [Event.LEFT_DOWN]
    outs, _ = run(e, [REST] * 6, t)
    assert events(outs) == [Event.LEFT_UP]


def test_real_opening_still_drops_promptly():
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 6 + [REST] * 6)
    up = next(i for i, o in enumerate(outs) if Event.LEFT_UP in o.events)
    assert (up - 16) / 30 <= 0.2                  # within ~0.15 s of opening


def test_opening_hand_after_grab_never_clicks():
    """Real data: fingers straightening after a drop read as a push -> stray click after
    every drag (which could open the file just moved)."""
    opening = [make_hand(push="imrp", amount=0.9 - 0.3 * k) for k in range(4)]
    half_open = make_hand(push="i", amount=0.5)   # index lags behind the others
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 10 + opening + [half_open] * 3
                  + [REST] * 10)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]


def test_fist_during_click_cooldown_still_grabs():
    """Real data: a push suppressed by the cooldown blocked the fist that followed."""
    curl = [make_hand(push="i", amount=0.3 * k) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + tap("i") + [REST] * 2 + curl + [FIST] * 8
                  + [REST] * 6)
    assert events(outs) == [Event.LEFT_CLICK, Event.LEFT_DOWN, Event.LEFT_UP]


def test_finger_dip_while_scrolling_is_not_a_right_click():
    """Real data: the middle finger dipped mid-scroll, the pose broke for a moment and
    the dip registered as a middle push."""
    scroll = [make_hand(ext=("i", "m"), dy=-2 * k) for k in range(20)]
    dip = [make_hand(ext=("i", "m"), push="m", amount=0.6, dy=-40)] * 4
    outs, _ = run(engine(), scroll + dip + scroll[::-1])
    assert events(outs) == []
    assert all(o.mode is Mode.SCROLL for o in outs[10:])


def test_brief_fist_flicker_does_not_grab():
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 2 + [REST] * 5)  # < FIST_ENTER_S
    assert events(outs) == []


def test_brief_finger_flick_during_grab_keeps_holding():
    """A single finger glitching open for a few frames (blur) is not letting go."""
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 10 + [POINTING] * 4 + [FIST] * 10
                  + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]
    assert all(o.mode is Mode.GRAB for o in outs[20:34])


def test_pointing_again_after_grab_drops():
    """Real data: after a fist the user went back to pointing (index out, others curled)
    and the grab never ended -- everything got dragged."""
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 10 + [POINTING] * 15)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]
    up = next(i for i, o in enumerate(outs) if Event.LEFT_UP in o.events)
    assert (up - 20) / 30 <= 0.4                  # within ~0.3 s of pointing


def test_pointing_then_fist_converts_index_push_into_one_grab():
    curl = [make_hand(ext=("i",), push="i", amount=0.3 * k) for k in range(1, 4)]
    outs, _ = run(engine(), [POINTING] * 10 + curl + [FIST] * 10 + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]   # never two downs
    assert outs[-4].mode is Mode.GRAB


def test_thumb_fold_then_fist_grabs_instead_of_right_click():
    curl = [make_hand(thumb_fold=k / 3) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + curl + [FIST] * 10 + [REST] * 6)
    assert Event.RIGHT_CLICK not in events(outs)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]


def test_right_click_fires_when_thumb_comes_back_out():
    outs, _ = run(engine(), [REST] * 10 + fold(frames_in=10) + [REST] * 3)
    assert events(outs) == [Event.RIGHT_CLICK]
    first_click = next(i for i, o in enumerate(outs) if Event.RIGHT_CLICK in o.events)
    assert first_click > 10 + 3 + 10   # after the fold was held, i.e. on the way back out
    assert any(o.press_started for o in outs[10:14])  # but aim was pinned as it folded


def test_hand_lost_during_grab_drops_after_grace():
    e = engine()
    outs, t = run(e, [REST] * 10 + [FIST] * 10 + [None] * 2)
    assert events(outs) == [Event.LEFT_DOWN]
    outs, _ = run(e, [None] * 25, t)
    assert events(outs) == [Event.LEFT_UP]


# --- Regressions from the user's real logged data -----------------------------------------

def test_relaxing_hand_middle_and_ring_together_is_not_right_click():
    """Real data: middle and ring shortened together (hand relaxing) -> false right click."""
    relax = [make_hand(push="mr", amount=0.5 * k / 3) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + relax + [relax[-1]] * 4 + relax[::-1] + [REST] * 3)
    assert events(outs) == []


def test_whole_hand_curling_is_not_a_click():
    curl = [make_hand(push="imrp", amount=0.5 * k / 3) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + curl + curl[::-1] + [REST] * 3)
    assert events(outs) == []


def test_relaxing_hand_with_thumb_drifting_in_is_not_right_click():
    # Relaxing curls the fingers a little and lets the thumb drift partway in (not a fold).
    relax = [make_hand(push="mr", amount=0.5 * k / 3, thumb_fold=0.13 * k) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + relax + [relax[-1]] * 4 + relax[::-1] + [REST] * 3)
    assert events(outs) == []


def test_fingers_pointing_at_camera_are_not_a_fist():
    """Real data: the 2D curl test read a sideways/forward-pointing open hand as a fist."""
    toward = make_hand(push="imrp", amount=0.7)
    assert not measure(toward, FS).fist
    outs, _ = run(engine(), [REST] * 10 + [toward] * 20)
    assert all(o.mode is not Mode.GRAB for o in outs)


def test_partly_relaxed_fist_keeps_holding():
    """Real data: the fist dropped while the hand was still mostly closed."""
    loose = make_hand(ext=("r",))          # one finger drifts open: still holding
    outs, _ = run(engine(), [REST] * 10 + [FIST] * 6 + [loose, FIST] * 10 + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]


def test_reach_separates_open_and_fist_like_real_data():
    open_reach = measure(REST, FS).reach
    fist_reach = measure(FIST, FS).reach
    assert min(open_reach) > 1.5 and max(fist_reach) < 1.0


def test_measurements_are_scale_invariant():
    near = measure(REST, FS)
    far_lm = REST.landmarks.copy()
    far_lm[:, :2] = 0.5 + (far_lm[:, :2] - 0.5) * 0.5   # same hand, half the size
    far_lm[:, 2] *= 0.5
    far = measure(type(REST)(far_lm, "Right", 0.99), FS)
    assert abs(near.index_ext - far.index_ext) < 1e-3


# --- Five-finger touch = left click ------------------------------------------------------

from .hands import make_touch, touch  # noqa: E402


def test_five_finger_touch_is_one_left_click_on_release():
    outs, _ = run(engine(), [REST] * 10 + touch() + [REST] * 5)
    assert events(outs) == [Event.LEFT_CLICK]
    clicked = next(i for i, o in enumerate(outs) if Event.LEFT_CLICK in o.events)
    assert any(o.mode is Mode.TOUCH for o in outs[:clicked])
    assert sum(o.press_started for o in outs) == 1       # aim pinned once, at the touch


def test_two_touches_are_two_clicks():
    outs, _ = run(engine(), [REST] * 10 + touch() + [REST] * 4 + touch() + [REST] * 5)
    assert events(outs) == [Event.LEFT_CLICK] * 2


def test_partial_touch_does_not_click():
    almost = [make_touch(k / 6) for k in range(1, 5)]       # fingertips never quite meet
    outs, _ = run(engine(), [REST] * 10 + almost + almost[::-1] + [REST] * 5)
    assert events(outs) == []


def test_fist_and_other_poses_are_not_a_touch():
    for hand in (REST, FIST, POINTING, make_hand(ext=("i", "m"))):
        assert not measure(hand, FS).touch


def test_touch_then_fist_grabs_instead_of_clicking():
    curl = [make_hand(ext=(), push=None)] * 8
    outs, _ = run(engine(), [REST] * 10 + touch()[:5] + curl + [REST] * 6)
    assert events(outs) == [Event.LEFT_DOWN, Event.LEFT_UP]


def test_long_held_touch_is_a_pose_not_a_click():
    outs, _ = run(engine(), [REST] * 10 + touch(frames_closed=60) + [REST] * 5)
    assert events(outs) == []


def test_fingertips_moving_like_a_push_become_one_touch_click():
    """Closing the fingertips moves the index tip like a push; it must click once (as a
    touch), not twice and not as a separate index tap."""
    push_then_touch = tap("i")[:3] + touch()
    outs, _ = run(engine(), [REST] * 10 + push_then_touch + [REST] * 6)
    assert events(outs) == [Event.LEFT_CLICK]


def test_opening_after_touch_never_adds_a_push_click():
    outs, _ = run(engine(), [REST] * 10 + touch() + [REST] * 15)
    assert events(outs) == [Event.LEFT_CLICK]


def test_quick_two_frame_touch_clicks():
    """Real data: touches often last just 2 camera frames."""
    outs, _ = run(engine(), [REST] * 10 + [make_touch(0.6), make_touch(1.0), make_touch(1.0),
                                           make_touch(0.6)] + [REST] * 5)
    assert events(outs) == [Event.LEFT_CLICK]


def test_one_frame_touch_blip_is_ignored():
    """Real data: single-frame 'touch' blips appeared during fast fist moves."""
    outs, _ = run(engine(), [REST] * 10 + [make_touch(1.0)] + [REST] * 5)
    assert events(outs) == []


def test_touch_is_told_apart_from_fist_by_where_the_fingertips_meet():
    """A fist tucks the fingertips into the palm; a touch meets out in front of it."""
    assert measure(make_touch(1.0), FS).touch_out > 0.6
    assert measure(FIST, FS).touch_out < 0.38


# --- Thumb + index pinch = double click ------------------------------------------------------------

from .hands import make_pinch, pinch  # noqa: E402


def test_pinch_double_clicks_once_as_it_closes():
    outs, _ = run(engine(), [REST] * 10 + pinch(frames_closed=10) + [REST] * 5)
    assert events(outs) == [Event.DOUBLE_CLICK]
    fired = next(i for i, o in enumerate(outs) if Event.DOUBLE_CLICK in o.events)
    assert fired <= 10 + 3 + 2                        # on closing, not after opening


def test_two_pinches_are_two_double_clicks():
    outs, _ = run(engine(), [REST] * 10 + pinch() + [REST] * 4 + pinch() + [REST] * 4)
    assert events(outs) == [Event.DOUBLE_CLICK] * 2


def test_pinch_measures_apart_from_touch_and_fist():
    assert measure(make_pinch(1.0), FS).pinch
    assert not measure(make_touch(1.0), FS).pinch     # all fingertips close: a touch
    assert not measure(FIST, FS).pinch
    assert not measure(REST, FS).pinch


def test_five_finger_touch_never_double_clicks():
    outs, _ = run(engine(), [REST] * 10 + touch() + [REST] * 5)
    assert events(outs) == [Event.LEFT_CLICK]


def test_index_bending_to_pinch_is_not_also_a_tap_click():
    bend = [make_hand(push="i", amount=0.2 * k) for k in range(1, 4)]
    outs, _ = run(engine(), [REST] * 10 + bend + pinch()[2:] + [REST] * 6)
    assert events(outs) == [Event.DOUBLE_CLICK]


# --- Joystick scrolling ----------------------------------------------------------------------------

def _scroll_hand(dy: float = 0.0, scale: float = 1.0):
    h = make_hand(ext=("i", "m"), dy=dy)
    if scale != 1.0:                                   # hand nearer/further from the camera
        lm = h.landmarks.copy()
        lm[:, :2] = 0.5 + (lm[:, :2] - 0.5) * scale
        h = type(h)(lm, "Right", 0.99)
    return h


def test_holding_above_the_scroll_line_keeps_scrolling_up():
    start = [_scroll_hand()] * 10                      # sets the line
    above = [_scroll_hand(dy=-60.0)] * 30              # held ~0.4 palm above it, still
    outs, _ = run(engine(), start + above)
    later = [o.scroll for o in outs[-20:]]
    assert sum(later) > 0 and sum(1 for v in later if v > 0) >= 15   # keeps going while still


def test_below_the_line_scrolls_down_and_back_on_the_line_stops():
    outs, _ = run(engine(), [_scroll_hand()] * 10 + [_scroll_hand(dy=60.0)] * 20
                  + [_scroll_hand()] * 25)
    assert sum(o.scroll for o in outs[10:30]) < 0
    assert all(o.scroll == 0 for o in outs[-12:])      # back at the line: stopped


def test_wobble_near_the_line_does_not_scroll():
    wobble = [_scroll_hand(dy=8.0 * ((-1) ** k)) for k in range(40)]
    outs, _ = run(engine(), [_scroll_hand()] * 10 + wobble)
    assert sum(abs(o.scroll) for o in outs) == 0


def test_returning_to_the_line_does_not_undo_the_scroll():
    """User report: scrolling didn't work properly -- with stroke-following, every return
    stroke scrolled back (recordings: 3,521 up vs 4,021 down)."""
    up = [_scroll_hand(dy=-6.0 * k) for k in range(1, 11)] + [_scroll_hand(dy=-60.0)] * 15
    back = [_scroll_hand(dy=-60.0 + 6.0 * k) for k in range(1, 11)]
    outs, _ = run(engine(), [_scroll_hand()] * 10 + up + back + [_scroll_hand()] * 10)
    assert sum(o.scroll for o in outs) > 300          # net progress, not cancelled
    assert all(o.scroll >= 0 for o in outs)            # never scrolled back down


def test_leaning_toward_the_camera_does_not_scroll():
    lean = [_scroll_hand(scale=1.0 + 0.01 * k) for k in range(15)]
    outs, _ = run(engine(), [_scroll_hand()] * 10 + lean + [lean[-1]] * 10)
    assert abs(sum(o.scroll for o in outs)) < 60


def test_scroll_reports_its_line_for_the_window():
    outs, _ = run(engine(), [_scroll_hand()] * 12)
    assert outs[-1].mode is Mode.SCROLL and outs[-1].scroll_anchor is not None
