"""Gesture recognition: hand landmarks -> mouse intents (click, drag, right click, scroll).

Clicks are "air taps": push a finger forward toward the camera and back, like tapping a
button. The click fires on the pull-back, and only if the push was brief: a push held for
longer (e.g. naturally pointing at the screen, which foreshortens the finger just like a
push) is a posture and never holds a button down. Dragging is done with a fist.
A webcam has no true depth, so a finger's forward extension is estimated from two cues:
  * foreshortening - a finger tipping toward the camera looks shorter in the image;
  * MediaPipe's relative z - the fingertip gets closer to the camera than its knuckle.
Each finger is compared with its own slowly-adapting resting baseline, so only a fairly
quick push counts, not gradual posture changes.

Grab: closing the hand into a fist holds the left button and the cursor follows the hand;
opening the hand drops. Works for anything draggable: icons, files, windows, sliders.
Because taps fire on pull-back, a fist (which often begins with one finger curling)
cancels the tap instead of misfiring.

Distances are in frame pixels divided by the palm length (wrist -> middle knuckle), so
thresholds hold at any distance from the camera.

Open vs. curled fingers use 3D "reach" (wrist -> fingertip distance including MediaPipe's
depth, over the 3D palm length). Tuned on real logged hand data: open fingers reach
1.6-2.5, a fist 0.6-0.9, with almost nothing between. A 2D test (tip nearer the wrist
than the middle joint) flickered ~45x more and misread sideways-pointing hands as fists.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .config import Config
from .cursor import EmaFilter
from .tracker import (INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_TIP, RING_TIP,
                      THUMB_TIP, WRIST, Hand)

RING_MCP = 13
PINKY_MCP = 17

Z_WEIGHT = 0.5            # how much MediaPipe's depth estimate counts vs. foreshortening
RELEASE_FACTOR = 0.5      # a push releases once its drop falls below threshold * this
MIN_BASELINE = 0.15       # numeric floor for the relative-drop division
WINNER_MARGIN = 0.15      # pushed finger must beat every other finger's drop by this much
                          # (real pushes beat them by 0.8+; a false one by 0.10)
TAP_MAX_S = 1.0           # a push held longer is a posture, not a click (real taps: 0.1-0.6 s)
GRAB_REFRACTORY_S = 0.4   # no new pushes this long after a grab or scroll ends
PUSH_RISE_MAX_S = 0.35    # a push must go from rest to pressed this fast (real: < 0.1 s);
                          # slow drifts that creep over the threshold aren't clicks
TAP_MOVE_MAX = 0.8        # a tap keeps the hand fairly still (real taps: knuckle moves <= 0.57
                          # palm lengths; reaching: 0.6-1.6), so moving the hand cancels it
SCROLL_EXIT_S = 0.3       # scrolling survives a brief break in the pose (finger dip)
BASELINE_ALPHA = 0.03     # resting-baseline EMA weight per frame (~1 s to adapt at 30 fps)
REBASE_AFTER_S = 0.5      # a partial dip held this long is a new resting pose, not a push
OPEN_REACH = 1.45         # 3D reach above this = finger open
CURLED_REACH = 1.3        # 3D reach below this = finger curled
FIST_ENTER_REACH = 1.2    # fist when every finger's reach is below this...
FIST_RELEASE_REACH = 1.5  # ...and it drops once two fingers reach beyond this
SCROLL_TIPS_APART = 0.5   # index & middle tips must be within this many palm lengths
SCROLL_ENTER_S = 0.15     # scroll pose must be held this long before scrolling starts
SCROLL_DEADZONE = 0.12    # joystick scroll: hand this close to where scrolling began
                          # (palm lengths) = no scrolling, so normal wobble doesn't scroll
SCROLL_RATE_GAIN = 2.0    # wheel units/s = scroll_speed * GAIN * offset beyond the dead zone
SCROLL_MAX_RATE = 4000.0  # wheel units/s (~33 notches/s)
LOST_GRACE_S = 0.25       # keep a held press through brief tracking dropouts
FIST_ENTER_S = 0.1        # fist held this long = grab
GRAB_OPEN_CONFIRM_S = 0.12  # hand must look open this long to drop: fast moves blur
                            # the fingers for a frame or two, which isn't letting go
BUD_SPREAD_ENTER = 0.45   # five-finger touch: every fingertip this close to the thumb tip
                          # (palm lengths, 3D; real touches: 0.26-0.44)...
BUD_OUT_MIN = 0.38        # ...and the fingertip cluster out in front of the palm centre
                          # (real touches 0.43-0.72; a fist tucks them in: median 0.25)
BUD_SPREAD_RELEASE = 0.6  # fingertips this far apart again = touch released -> click
BUD_ENTER_FRAMES = 2      # consecutive frames (real touches can last just 2 frames)
BUD_MAX_S = 1.5           # touches held longer are a pose, not a click
THUMB_OUT = 1.0           # thumb tip -> little-finger knuckle (3D / palm) when the thumb is
                          # out; open hand: 1.1-1.47. But it is often tucked in (0.2-0.9)
                          # while pointing or scrolling, so a fold is a MOVEMENT, not a pose:
THUMB_IN = 0.65           # folded across the palm...
FOLD_RISE_MAX_S = 0.4     # ...within this long of last being out...
FINGERS_STILL = 0.3       # ...while no finger's reach changes more than this (a fist, touch,
                          # or switching to pointing/scrolling moves the fingers too)
THUMB_RELEASE = 0.9       # thumb back out: right click
PINCH_ENTER = 0.25        # double click: thumb and index tips this close (3D / palm)...
PINCH_OTHERS_MIN = 0.45   # ...while the other fingertips stay this far from the thumb (and
                          # not a fist). On the user's real data this fired 0 false pinches;
                          # thumb-index alone fired 102 (fists, five-finger touches)
PINCH_RELEASE = 0.4       # fingers apart again before the next pinch can double-click
PINCH_FRAMES = 2          # consecutive frames (~70 ms)
GRAB_POINT_CONFIRM_S = 0.3  # just the index open (back to pointing) this long also drops;
                            # single-finger blur glitches in real grabs lasted <= 0.16 s
GRAB_LOST_GRACE_S = 0.6  # tracking can drop out briefly while a fist moves fast


class Mode(Enum):
    NO_HAND = "No hand"
    POINT = "Point"
    LEFT_PRESS = "Left press"
    RIGHT_PRESS = "Thumb fold"  # thumb folded into the palm: right click on release
    GRAB = "Grab"           # fist: left button held, cursor follows the hand
    TOUCH = "Touch"         # all five fingertips together: left click on release
    PINCH = "Pinch"         # thumb + index tips together: double click (fires on close)
    SCROLL = "Scroll"


class Event(Enum):
    LEFT_CLICK = "left_click"    # index tap or five-finger touch (twice = double click)
    DOUBLE_CLICK = "double_click"  # thumb + index pinch
    RIGHT_CLICK = "right_click"  # thumb folded into the palm and back out
    LEFT_DOWN = "left_down"      # fist closed: grab
    LEFT_UP = "left_up"          # hand opened: drop


@dataclass
class GestureOutput:
    mode: Mode
    events: list[Event] = field(default_factory=list)
    press_started: bool = False  # a press or grab began this frame (pins the cursor's aim)
    releasing: bool = False      # grab: the hand looks open, confirming the drop -- hold still
    scroll: int = 0          # wheel units (120 = one notch), positive = up
    scroll_anchor: tuple[float, float] | None = None  # (y, dead-zone half height), frame fractions
    left_push: float = 0.0   # index finger's relative drop from its resting extension
    right_push: float = 0.0  # middle finger's relative drop from its resting extension


@dataclass
class Features:
    index_ext: float         # forward-extension signal; smaller = pushed toward the camera
    middle_ext: float
    ring_ext: float          # ring moves with the middle when the hand relaxes: a reference
    scroll_pose: bool
    scroll_y: float          # midpoint of index & middle tips, fraction of frame height
    palm_h: float            # palm length as a fraction of frame height
    reach: tuple[float, float, float, float]  # 3D wrist->tip / palm: index, middle, ring, pinky
    knuckle: tuple[float, float]  # index knuckle position in palm lengths (hand location)
    touch_spread: float      # farthest fingertip from the thumb tip, 3D / palm (0 = touching)
    touch_out: float         # fingertip cluster's distance from the palm centre, 3D / palm
    thumb_in: float          # thumb tip -> little-finger knuckle, 3D / palm (small = folded)
    pinch_gap: float         # thumb tip -> index tip, 3D / palm
    others_gap: float        # nearest of the middle/ring/pinky tips to the thumb tip, 3D / palm

    @property
    def curled(self) -> int:
        return sum(r < CURLED_REACH for r in self.reach)

    @property
    def fist(self) -> bool:
        return max(self.reach) < FIST_ENTER_REACH

    @property
    def touch(self) -> bool:
        """All five fingertips together, out in front of the palm (a fist tucks them in)."""
        return self.touch_spread < BUD_SPREAD_ENTER and self.touch_out > BUD_OUT_MIN

    @property
    def pinch(self) -> bool:
        """Thumb and index tips together while the other fingertips stay clear."""
        return (self.pinch_gap < PINCH_ENTER and self.others_gap > PINCH_OTHERS_MIN
                and not self.fist)

    @property
    def pointing_out(self) -> bool:
        """At least one finger clearly open (e.g. pointing again after a grab)."""
        return max(self.reach) > FIST_RELEASE_REACH

    @property
    def opened(self) -> bool:
        """At least two fingers clearly open: ends a grab. (Pointing alone doesn't.)"""
        return sorted(self.reach)[-2] > FIST_RELEASE_REACH


def measure(hand: Hand, frame_size: tuple[int, int]) -> Features:
    # Normalized x and y are fractions of different frame sides, so convert to pixels first.
    pts = hand.landmarks[:, :2] * frame_size
    # MediaPipe z is on roughly the same scale as normalized x.
    z = hand.landmarks[:, 2] * frame_size[0]
    pts3 = np.column_stack([pts, z])

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    palm = max(dist(WRIST, MIDDLE_MCP), 1e-6)
    palm3 = max(float(np.linalg.norm(pts3[WRIST] - pts3[MIDDLE_MCP])), 1e-6)

    def extension(tip: int, mcp: int) -> float:
        depth = (z[tip] - z[mcp]) / palm  # negative when the tip is nearer the camera
        return dist(mcp, tip) / palm + Z_WEIGHT * float(depth)

    def reach(tip: int) -> float:
        return float(np.linalg.norm(pts3[tip] - pts3[WRIST])) / palm3

    r_index, r_middle, r_ring, r_pinky = (reach(t) for t in
                                          (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP))
    scroll_pose = (
        r_index > OPEN_REACH and r_middle > OPEN_REACH
        and r_ring < CURLED_REACH and r_pinky < CURLED_REACH
        and dist(INDEX_TIP, MIDDLE_TIP) / palm < SCROLL_TIPS_APART
    )
    return Features(
        index_ext=extension(INDEX_TIP, INDEX_MCP),
        middle_ext=extension(MIDDLE_TIP, MIDDLE_MCP),
        ring_ext=extension(RING_TIP, RING_MCP),
        scroll_pose=scroll_pose,
        scroll_y=float((pts[INDEX_TIP, 1] + pts[MIDDLE_TIP, 1]) / 2) / frame_size[1],
        palm_h=palm / frame_size[1],
        reach=(r_index, r_middle, r_ring, r_pinky),
        knuckle=(float(pts[INDEX_MCP, 0]) / palm, float(pts[INDEX_MCP, 1]) / palm),
        touch_spread=max(float(np.linalg.norm(pts3[t] - pts3[THUMB_TIP]))
                         for t in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)) / palm3,
        thumb_in=float(np.linalg.norm(pts3[THUMB_TIP] - pts3[PINKY_MCP])) / palm3,
        pinch_gap=float(np.linalg.norm(pts3[THUMB_TIP] - pts3[INDEX_TIP])) / palm3,
        others_gap=min(float(np.linalg.norm(pts3[THUMB_TIP] - pts3[t]))
                       for t in (MIDDLE_TIP, RING_TIP, PINKY_TIP)) / palm3,
        touch_out=float(np.linalg.norm(
            pts3[[THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]].mean(axis=0)
            - pts3[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0))) / palm3,
    )


class PushTracker:
    """Tracks one finger's resting extension and reports how far it is pushed from it."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._baseline: float | None = None
        self._dip_since: float | None = None
        self.rest_at = -1e9   # last time the finger was at rest (drop below release level)
        self.rest_reach = 0.0  # its 3D reach while at rest: curled fingers can't push

    def rebase(self, ext: float) -> None:
        """Adopt the current extension as the finger's resting pose."""
        self._baseline = ext
        self._dip_since = None

    def rose_fast(self, now: float) -> bool:
        """True if the finger was at rest within PUSH_RISE_MAX_S: a deliberate push."""
        return now - self.rest_at <= PUSH_RISE_MAX_S

    def update(self, ext: float, reach: float, now: float, threshold: float,
               frozen: bool) -> float:
        """Return the relative drop (0 = at rest, 0.3 = 30% shorter/closer than at rest).

        Always 0 for a finger that was curled at rest (3D reach): only an extended finger
        can be pushed. Reach, unlike 2D extension, stays long for a finger pointing at
        the camera, so that pose remains pushable.
        """
        if self._baseline is None:
            self._baseline = ext
            self.rest_reach = reach
        base = self._baseline
        drop = (base - ext) / base if base > MIN_BASELINE else 0.0
        if drop < threshold * RELEASE_FACTOR:
            self.rest_at = now
            if not frozen:
                self.rest_reach = reach
        if self.rest_reach < OPEN_REACH:
            drop = 0.0
        if frozen:
            return drop

        if ext > base:
            # Pushes only ever shorten a finger, so rest is its most extended pose:
            # follow straightening immediately (e.g. the hand opening after a fist).
            self._baseline = ext
            self._dip_since = None
        elif drop < threshold * RELEASE_FACTOR:
            # At rest: let the baseline follow slow posture changes.
            self._baseline = base + BASELINE_ALPHA * (ext - base)
            self._dip_since = None
        else:
            # Partway down but not pressed: if it stays here, it's a new resting pose.
            if self._dip_since is None:
                self._dip_since = now
            elif now - self._dip_since > REBASE_AFTER_S:
                self._baseline = ext
                self._dip_since = None
        return drop


class GestureEngine:
    """Stateful recognizer. Feed it one frame at a time with a monotonic timestamp."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._scroll_filter = EmaFilter(cfg.smoothing_alpha)
        self._index = PushTracker()
        self._middle = PushTracker()
        self._ring = PushTracker()
        self._last_press = -1e9
        self.reset()

    def reset(self) -> None:
        """Forget the current hand. The click cooldown deliberately survives."""
        self._held: Mode | None = None       # LEFT_PRESS / RIGHT_PRESS / GRAB while held
        self._suppressed = False             # a push that must not click (cooldown)
        self._fist_since: float | None = None
        self._touch_frames = 0
        self._pinch_frames = 0
        self._open_since: float | None = None
        self._point_since: float | None = None
        self._press_since = -1e9
        self._press_knuckle = (0.0, 0.0)
        self._thumb_rest_at = -1e9            # last time the thumb was out
        self._thumb_rest_reach = (0.0, 0.0, 0.0, 0.0)  # finger reaches at that moment
        self._no_press_until = -1e9          # refractory period after a grab ends
        self._last_seen = -1e9
        self._scroll_since: float | None = None
        self._scrolling = False
        self._scroll_lost_since: float | None = None
        self._scroll_anchor: float | None = None   # hand height (frame y) when scrolling began
        self._scroll_t: float | None = None
        self._scroll_accum = 0.0
        self._scroll_filter.reset()
        self._index.reset()
        self._middle.reset()
        self._ring.reset()

    def update(self, hand: Hand | None, frame_size: tuple[int, int], now: float) -> GestureOutput:
        if hand is None:
            return self._on_lost(now)
        self._last_seen = now
        f = measure(hand, frame_size)
        threshold = self.cfg.push_threshold
        frozen = self._held is not None
        left = self._index.update(f.index_ext, f.reach[0], now, threshold, frozen)
        right = self._middle.update(f.middle_ext, f.reach[1], now, threshold, frozen)
        ring = self._ring.update(f.ring_ext, f.reach[2], now, threshold, frozen)
        out = GestureOutput(Mode.POINT, left_push=left, right_push=right)
        if f.thumb_in > THUMB_OUT and self._held is not Mode.RIGHT_PRESS:
            self._thumb_rest_at = now
            self._thumb_rest_reach = f.reach

        fist = f.fist
        if fist:
            if self._fist_since is None:
                self._fist_since = now
        else:
            self._fist_since = None
        touch = f.touch
        self._touch_frames = self._touch_frames + 1 if touch else 0
        self._pinch_frames = self._pinch_frames + 1 if f.pinch else 0

        # 1. A grab (fist) only ends when the hand clearly opens -- and stays open, so a
        #    blurry frame or two during a fast drag doesn't drop what you're carrying.
        if self._held is Mode.GRAB:
            if not f.opened:
                self._open_since = None
            elif self._open_since is None:
                self._open_since = now
            if not f.pointing_out:
                self._point_since = None
            elif self._point_since is None:
                self._point_since = now
            if ((self._open_since is not None and now - self._open_since >= GRAB_OPEN_CONFIRM_S)
                    or (self._point_since is not None
                        and now - self._point_since >= GRAB_POINT_CONFIRM_S)):
                self._open_since = self._point_since = None
                out.events.append(Event.LEFT_UP)
                self._end_touch_or_grab(f, now)
            else:
                out.mode = Mode.GRAB
                out.releasing = self._open_since is not None or self._point_since is not None
            return out

        # 1b. A pinch double-clicked when it closed; it only ends when the fingers part.
        if self._held is Mode.PINCH:
            if f.pinch_gap > PINCH_RELEASE:
                self._end_touch_or_grab(f, now)
            else:
                out.mode = Mode.PINCH
            return out

        # 1c. Thumb + index pinch: double click at once, overriding a push that the index
        #     bending toward the thumb may have started.
        if (self._pinch_frames >= PINCH_FRAMES
                and self._held not in (Mode.GRAB, Mode.TOUCH)):
            pinned = self._held is not None and not self._suppressed  # aim already pinned
            self._held = out.mode = Mode.PINCH
            self._suppressed = False
            out.events.append(Event.DOUBLE_CLICK)
            out.press_started = not pinned
            return out

        # 2. A five-finger touch clicks when the fingertips part again -- unless the hand
        #    closes into a fist instead (grab), or it was held so long it's just a pose.
        if self._held is Mode.TOUCH:
            if fist:
                self._held = out.mode = Mode.GRAB
                self._suppressed = False
                out.events.append(Event.LEFT_DOWN)   # aim was pinned when the touch began
                return out
            if f.touch_spread > BUD_SPREAD_RELEASE:
                if not self._suppressed and now - self._press_since <= BUD_MAX_S:
                    out.events.append(Event.LEFT_CLICK)
                self._end_touch_or_grab(f, now)
                self._suppressed = False
                return out
            out.mode = Mode.TOUCH if not self._suppressed else Mode.POINT
            return out

        # 3. A push in progress. Curling into a fist turns it into a grab (a fist often
        #    starts with one finger curling). Otherwise it's a tap: it clicks when the
        #    finger is pulled back, if that happened within TAP_MAX_S. A longer push is a
        #    posture (e.g. pointing at the camera): adopt it as rest, never click.
        if self._held is not None:
            if touch:
                # Bringing the fingertips together moves them like a push: it's a touch.
                pinned = not self._suppressed
                self._held = out.mode = Mode.TOUCH
                self._suppressed = False
                self._press_since = now
                out.press_started = not pinned
                return out
            if fist:
                pinned = not self._suppressed       # a clean press already pinned the aim
                self._held = out.mode = Mode.GRAB
                self._suppressed = False
                out.events.append(Event.LEFT_DOWN)
                out.press_started = not pinned
                return out
            moved = float(np.hypot(f.knuckle[0] - self._press_knuckle[0],
                                   f.knuckle[1] - self._press_knuckle[1]))
            if self._held is Mode.RIGHT_PRESS:
                return self._update_fold(f, now, moved, out)
            tracker, ext, drop = self._index, f.index_ext, left
            if moved > TAP_MOVE_MAX:
                # The whole hand is moving: not a tap. Let the cursor follow again.
                tracker.rebase(ext)
                self._held = None
                self._suppressed = False
                return out
            if drop < threshold * RELEASE_FACTOR:
                if not self._suppressed and now - self._press_since <= TAP_MAX_S:
                    out.events.append(Event.LEFT_CLICK)
                self._held = None
                self._suppressed = False
            elif now - self._press_since > TAP_MAX_S:
                tracker.rebase(ext)
                self._held = None
                self._suppressed = False
                return out
            else:
                out.mode = self._held if not self._suppressed else Mode.POINT
                return out

        # 4. Closing the hand into a fist grabs (left button down, cursor follows the hand).
        if fist and self._fist_since is not None and now - self._fist_since >= FIST_ENTER_S:
            self._held = out.mode = Mode.GRAB
            out.events.append(Event.LEFT_DOWN)
            out.press_started = True
            return out

        # 5. All five fingertips together: a touch (clicks on release).
        if self._touch_frames >= BUD_ENTER_FRAMES:
            self._held = out.mode = Mode.TOUCH
            self._press_since = now
            if now - self._last_press < self.cfg.click_cooldown:
                self._suppressed = True
                out.mode = Mode.POINT
                return out
            self._last_press = now
            out.press_started = True
            return out

        # 6. Scroll pose (index + middle up together) takes priority over new presses.
        if self._update_scroll(f, now, out):
            return out

        if now < self._no_press_until:
            return out

        # 7. Thumb folded quickly into the palm while the fingers stay put: right click
        #    (fires when the thumb comes back out).
        if (f.thumb_in < THUMB_IN and now - self._thumb_rest_at <= FOLD_RISE_MAX_S
                and self._fingers_still(f, self._thumb_rest_reach)):
            return self._start_press(Mode.RIGHT_PRESS, f, now, out)

        # 8. Index push: it must be isolated -- the index moves clearly more than the other
        #    fingers (the ring follows the middle whenever the hand relaxes or curls, and
        #    whole-hand motion moves them all together).
        if (left > threshold and left - max(right, ring) > WINNER_MARGIN
                and self._index.rose_fast(now)):
            return self._start_press(Mode.LEFT_PRESS, f, now, out)
        return out

    def _start_press(self, which: Mode, f: Features, now: float,
                     out: GestureOutput) -> GestureOutput:
        self._held = which
        self._press_since = now
        self._press_knuckle = f.knuckle
        if now - self._last_press < self.cfg.click_cooldown:
            self._suppressed = True
            return out
        self._last_press = now
        out.mode = which
        out.press_started = True
        return out

    @staticmethod
    def _fingers_still(f: Features, before: tuple[float, float, float, float]) -> bool:
        return max(abs(a - b) for a, b in zip(f.reach, before)) <= FINGERS_STILL

    def _update_fold(self, f: Features, now: float, moved: float,
                     out: GestureOutput) -> GestureOutput:
        """A thumb fold in progress: right click when the thumb comes back out, unless
        the hand moved, the fingers changed pose, or it was held too long."""
        if moved > TAP_MOVE_MAX or not self._fingers_still(f, self._thumb_rest_reach):
            self._held = None
            self._suppressed = False
            return out
        if f.thumb_in > THUMB_RELEASE:
            if not self._suppressed and now - self._press_since <= TAP_MAX_S:
                out.events.append(Event.RIGHT_CLICK)
            self._held = None
            self._suppressed = False
            return out
        if now - self._press_since > TAP_MAX_S:
            self._held = None
            self._suppressed = False
            return out
        out.mode = Mode.RIGHT_PRESS if not self._suppressed else Mode.POINT
        return out

    def _end_touch_or_grab(self, f: Features, now: float) -> None:
        """Opening the hand straightens every finger, which can look like a push
        releasing: start fresh and ignore pushes briefly, so letting go never clicks."""
        self._held = None
        for tracker, ext in ((self._index, f.index_ext), (self._middle, f.middle_ext),
                             (self._ring, f.ring_ext)):
            tracker.rebase(ext)
        self._no_press_until = now + GRAB_REFRACTORY_S

    def _update_scroll(self, f: Features, now: float, out: GestureOutput) -> bool:
        if not f.scroll_pose:
            if self._scrolling:
                # Ride out brief breaks (a finger dipping mid-scroll) instead of exiting --
                # and letting that dip read as a click.
                if self._scroll_lost_since is None:
                    self._scroll_lost_since = now
                if now - self._scroll_lost_since < SCROLL_EXIT_S:
                    out.mode = Mode.SCROLL
                    return True
                self._no_press_until = now + GRAB_REFRACTORY_S
            self._scroll_since = None
            self._scroll_lost_since = None
            self._scrolling = False
            self._scroll_anchor = None
            self._scroll_t = None
            self._scroll_accum = 0.0
            self._scroll_filter.reset()
            return False
        self._scroll_lost_since = None

        if self._scroll_since is None:
            self._scroll_since = now
        if not self._scrolling and now - self._scroll_since < SCROLL_ENTER_S:
            return False  # pose not stable yet; keep pointing
        self._scrolling = True
        out.mode = Mode.SCROLL

        # Joystick: where the hand was when scrolling began is "stop". Held above it the
        # page scrolls up, below it down, faster the further away. (Following hand strokes
        # instead made every return stroke scroll back: the user's recordings scrolled
        # 3,521 units up and 4,021 down in 8 s, almost nothing net.)
        y = float(self._scroll_filter.update(np.array([f.scroll_y]))[0])
        if self._scroll_anchor is None:
            self._scroll_anchor = y
            self._scroll_t = now
        dt = min(max(now - (self._scroll_t or now), 0.0), 0.1)
        self._scroll_t = now
        offset = (self._scroll_anchor - y) / max(f.palm_h, 1e-6)   # palm lengths, + = above
        beyond = abs(offset) - SCROLL_DEADZONE
        if beyond > 0:
            rate = min(self.cfg.scroll_speed * SCROLL_RATE_GAIN * beyond, SCROLL_MAX_RATE)
            self._scroll_accum += float(np.sign(offset)) * rate * dt

        whole = int(self._scroll_accum)
        self._scroll_accum -= whole
        out.scroll = whole
        out.scroll_anchor = (self._scroll_anchor, SCROLL_DEADZONE * f.palm_h)
        return True

    def _on_lost(self, now: float) -> GestureOutput:
        grace = GRAB_LOST_GRACE_S if self._held is Mode.GRAB else LOST_GRACE_S
        if self._held is not None and now - self._last_seen < grace:
            # Brief dropout: keep holding so a drag survives a missed frame or two.
            return GestureOutput(self._held if not self._suppressed else Mode.POINT)
        out = GestureOutput(Mode.NO_HAND)
        if self._held is Mode.GRAB:
            out.events.append(Event.LEFT_UP)    # an unfinished tap is just dropped
        self.reset()
        return out
