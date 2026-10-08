import numpy as np
import pytest

from airmouse.cursor import ClickStabilizer, EmaFilter, active_region, map_to_screen

FRAME = (1280, 720)
SCREEN = (1920, 1200)
REGION = active_region(FRAME, SCREEN, speed=2.0, center_y=0.5)


def screen_xy(x: float, y: float) -> np.ndarray:
    return map_to_screen(np.array([x, y]), FRAME, REGION, SCREEN)


def test_region_is_centred_with_screen_aspect():
    x0, y0, x1, y1 = REGION
    assert (x0, x1) == (320, 960)                         # width = frame / speed
    assert (y1 - y0) / (x1 - x0) == pytest.approx(1200 / 1920)
    assert (y0 + y1) / 2 == 360


def test_equal_hand_motion_gives_equal_cursor_motion_both_axes():
    x0, y0, x1, y1 = REGION
    assert 1920 / (x1 - x0) == pytest.approx(1200 / (y1 - y0))


def test_higher_speed_means_smaller_region():
    slow = active_region(FRAME, SCREEN, 1.5, 0.5)
    fast = active_region(FRAME, SCREEN, 3.0, 0.5)
    assert fast[2] - fast[0] < slow[2] - slow[0]


@pytest.mark.parametrize("center_y", [0.0, 0.25, 0.75, 1.0])
def test_region_stays_inside_frame(center_y):
    x0, y0, x1, y1 = active_region(FRAME, SCREEN, 1.0, center_y)  # biggest region
    assert 0 <= x0 < x1 <= FRAME[0] and 0 <= y0 < y1 <= FRAME[1]


def test_center_y_moves_region_up():
    high = active_region(FRAME, SCREEN, 2.5, 0.35)
    low = active_region(FRAME, SCREEN, 2.5, 0.6)
    assert high[1] < low[1]


def test_frame_center_maps_to_screen_center():
    np.testing.assert_allclose(screen_xy(0.5, 0.5), (959.5, 599.5))


@pytest.mark.parametrize("norm, expected", [
    ((320 / 1280, 160 / 720), (1, 1)),                 # active-region corner -> screen corner
    ((960 / 1280, 560 / 720), (1918, 1198)),
    ((0.0, 0.0), (1, 1)),                               # outside the region clamps
    ((1.0, 1.0), (1918, 1198)),
])
def test_region_edges_map_to_screen_edges(norm, expected):
    np.testing.assert_allclose(screen_xy(*norm), expected)


def test_hand_can_never_reach_failsafe_corners():
    corners = {(0, 0), (0, SCREEN[1] - 1), (SCREEN[0] - 1, 0), (SCREEN[0] - 1, SCREEN[1] - 1)}
    for x in np.linspace(-0.2, 1.2, 15):
        for y in np.linspace(-0.2, 1.2, 15):
            sx, sy = np.round(screen_xy(x, y)).astype(int)
            assert (sx, sy) not in corners
            assert 0 < sx < SCREEN[0] - 1 and 0 < sy < SCREEN[1] - 1


def test_ema_first_sample_passes_through_then_smooths():
    f = EmaFilter(0.25)
    assert f.update(np.array([100.0]))[0] == 100
    assert f.update(np.array([200.0]))[0] == pytest.approx(125)
    assert f.update(np.array([200.0]))[0] == pytest.approx(143.75)


def test_ema_reset_forgets_history():
    f = EmaFilter(0.25)
    f.update(np.array([0.0]))
    f.reset()
    assert f.update(np.array([500.0]))[0] == 500


def test_ema_converges_to_constant_input():
    f = EmaFilter(0.25)
    out = f.update(np.array([0.0, 0.0]))
    for _ in range(100):
        out = f.update(np.array([42.0, -7.0]))
    np.testing.assert_allclose(out, (42, -7), atol=1e-6)


KNUCKLE = np.array([500.0, 500.0])


def _warm(s: ClickStabilizer, n: int = 10) -> float:
    """Feed a slow rightward drift (x = 100..109); returns the next timestamp."""
    t = 0.0
    for k in range(n):
        s.update(np.array([100.0 + k, 100.0]), KNUCKLE, t, False, False, False)
        t += 1 / 30
    return t


def test_stabilizer_passes_through_when_idle():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    out = s.update(np.array([300.0, 300.0]), KNUCKLE, t, False, False, False)
    np.testing.assert_array_equal(out, (300, 300))


def test_stabilizer_snaps_back_on_press_and_ignores_fingertip_motion():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    pinned = s.update(np.array([115.0, 120.0]), KNUCKLE, t, True, True, True)
    assert pinned[0] <= 104  # where the cursor was ~0.2 s before the press
    # Finger pushes deeper: fingertip moves 100 px, knuckle doesn't -> no drag.
    deeper = s.update(np.array([115.0, 220.0]), KNUCKLE, t + 0.03, False, True, True)
    np.testing.assert_array_equal(deeper, pinned)


def test_stabilizer_knuckle_motion_while_held_becomes_drag():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    pinned = s.update(np.array([115.0, 120.0]), KNUCKLE, t, True, True, True)
    small = s.update(np.array([0.0, 0.0]), KNUCKLE + (20, 0), t + 0.03, False, True, True)
    np.testing.assert_array_equal(small, pinned)             # inside the deadzone
    drag = s.update(np.array([0.0, 0.0]), KNUCKLE + (80, 10), t + 0.06, False, True, True)
    np.testing.assert_array_equal(drag, pinned + (80, 10))   # follows the hand, not the tip


def test_stabilizer_non_draggable_press_ignores_hand_motion():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    pinned = s.update(np.array([115.0, 120.0]), KNUCKLE, t, True, True, False)
    out = s.update(np.array([300.0, 300.0]), KNUCKLE + (200, 0), t + 0.03, False, True, False)
    np.testing.assert_array_equal(out, pinned)


def test_stabilizer_releases_after_hold_time():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    pinned = s.update(np.array([115.0, 120.0]), KNUCKLE, t, True, True, True)
    still = s.update(np.array([116.0, 120.0]), KNUCKLE, t + 0.05, False, False, True)
    np.testing.assert_array_equal(still, pinned)            # just released: still pinned
    s.update(np.array([117.0, 120.0]), KNUCKLE, t + 0.25, False, False, True)
    free = s.update(np.array([117.0, 120.0]), KNUCKLE, t + 0.25 + s.HANDOVER_S, False, False,
                    True)
    np.testing.assert_array_equal(free, (117, 120))         # handed back to the fingertip


def test_grab_follows_hand_immediately_and_glides_back_after_drop():
    s = ClickStabilizer(0.2, 30, 0.15)
    t = _warm(s)
    pinned = s.update(np.array([115.0, 120.0]), KNUCKLE, t, True, True, False, grab=True)
    # Closing the fist moves the fingertip a lot; the knuckle barely moves.
    small = s.update(np.array([115.0, 400.0]), KNUCKLE + (5, 0), t + 0.03, False, True, False,
                     grab=True)
    np.testing.assert_array_equal(small, pinned + (5, 0))   # no deadzone for a grab
    carried = s.update(np.array([0.0, 0.0]), KNUCKLE + (300, 50), t + 0.06, False, True, False,
                       grab=True)
    np.testing.assert_array_equal(carried, pinned + (300, 50))

    # Open the hand: the fingertip target is now ~330 px above where the fist was.
    tip = carried + (0, -330)
    t2 = t + 0.1
    assert np.array_equal(s.update(tip, KNUCKLE, t2, False, False, False), carried)
    t2 += s.release_hold
    path = [s.update(tip, KNUCKLE, t2 + k * 0.03, False, False, False) for k in range(15)]
    steps = [np.linalg.norm(b - a) for a, b in zip(path, path[1:])]
    np.testing.assert_array_equal(path[0], carried)          # no jump at handover...
    assert max(steps) < 60                                   # ...a glide, not a 330 px snap
    np.testing.assert_array_equal(path[-1], tip)             # and it ends on the fingertip


# --- Taskbar zone mapping ----------------------------------------------------------------------

BAR = (0, 0, 0, 60)                                   # 60 px taskbar at the bottom


def sy(v: float, zone: float = 0.15, bands=BAR) -> float:
    region = (0.0, 0.0, 1000.0, 1000.0)
    return float(map_to_screen(np.array([0.5, v]), (1000, 1000), region, SCREEN, bands, zone)[1])


def test_bottom_zone_of_hand_area_maps_onto_the_taskbar():
    assert sy(0.85) == pytest.approx(1200 - 60)          # zone starts exactly at the taskbar
    assert 1200 - 60 < sy(0.92) < 1200 - 2                # inside the zone: on the taskbar
    assert sy(1.0) == pytest.approx(1198)                 # bottom edge, off the failsafe corner


def test_taskbar_gets_three_times_the_hand_room():
    plain = (sy(1.0, 0.0) - sy(0.95, 0.0))                # 5% of hand travel, no zone
    zoned = (sy(1.0) - sy(0.95))
    assert zoned < plain / 2.5                            # far less cursor motion per hand motion


def test_mapping_stays_monotonic_and_in_bounds_with_zones():
    vals = [sy(v, bands=(80, 40, 0, 60)) for v in np.linspace(-0.2, 1.2, 141)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert 1 <= min(vals) and max(vals) <= 1198


def test_taskbar_on_top_is_handled_too():
    top = (0, 60, 0, 0)
    assert sy(0.15, bands=top) == pytest.approx(60)
    assert sy(0.05, bands=top) < 60
