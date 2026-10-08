"""Guided gesture recording, for tuning detection on a real user's hand.

Run:  .venv\\Scripts\\python.exe -m airmouse.record   (quit AirMouse first: it needs the camera)
      .venv\\Scripts\\python.exe -m airmouse.record touch_tap touch_double   (only those steps)

Walks through each gesture with on-screen prompts (~105 s) and saves per-frame hand
landmarks (image + 3D world coordinates) with the prompted gesture as a label to
logs/recording-*.npz. Never controls the mouse; no video is saved. Q / Esc stops early;
whatever was captured is saved however the recording ends.
"""

import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyautogui

from .camera import Camera
from .config import LOG_DIR, Config
from .cursor import active_region
from .overlay import draw_active_region, draw_hand
from .settings import load_config
from .tracker import HandTracker

WINDOW = "AirMouse - gesture recording"
READY_S = 3.0

# (label, seconds, instruction)
STEPS: list[tuple[str, float, str]] = [
    ("open", 5, "Hold your hand OPEN, palm to the camera, fingers up. Keep still."),
    ("point", 5, "POINT with your index finger (other fingers curled). Keep still."),
    ("index_push", 10, "INDEX finger: push forward and back like clicking, about 1 per second."),
    ("thumb_fold", 10, "THUMB: fold it into your palm and back out, about once a second."),
    ("fist_cycle", 12, "Close into a FIST, then OPEN again. Slowly, about every 2 seconds."),
    ("fist_move", 8, "Make a FIST and move it around slowly. Keep it closed."),
    ("fist_fast", 8, "Make a FIST and move it around FAST, like a quick drag. Keep it closed."),
    ("pinch", 10, "PINCH: touch your THUMB and INDEX fingertips together, then open. ~1 per second."),
    ("touch_tap", 10, "Touch all FIVE fingertips together, then open. About once a second."),
    ("touch_double", 10, "DOUBLE: touch-open-touch quickly (like a double click), pause, repeat."),
    ("scroll", 8, "INDEX + MIDDLE up together, others curled. Move the hand up and down."),
    ("point_move", 10, "POINT and move your hand slowly over the whole green area."),
    ("reach", 10, "POINT at the very TOP of the green area, then the very BOTTOM. Repeat."),
]


@dataclass
class Recording:
    times: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    present: list[bool] = field(default_factory=list)
    image_lm: list[np.ndarray] = field(default_factory=list)
    world_lm: list[np.ndarray] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)
    aborted: bool = True

    def save(self) -> None:
        if not self.times:
            print("Nothing was recorded.")
            return
        out_dir = LOG_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / time.strftime("recording-%Y%m%d-%H%M%S.npz")
        np.savez_compressed(
            path, t=np.array(self.times), label=np.array(self.labels),
            present=np.array(self.present), landmarks=np.stack(self.image_lm),
            world=np.stack(self.world_lm), frame_size=np.array(self.frame_size),
            aborted=np.array(self.aborted),
        )
        kind = "Partial recording" if self.aborted else "Recording"
        print(f"{kind} saved: {path} ({len(self.times)} frames)")


def draw_prompt(frame: np.ndarray, title: str, text: str, progress: float,
                color: tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 110), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, title, (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                cv2.LINE_AA)
    cv2.putText(frame, text, (20, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1,
                cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 8), (int(w * progress), h), color, -1)


def run_steps(cam: Camera, tracker: HandTracker, cfg: Config, rec: Recording,
              steps: list[tuple[str, float, str]]) -> bool:
    """Prompt through every step, appending frames to `rec`. Returns False if stopped early."""
    blank = np.full((21, 3), np.nan, np.float32)
    x0, y0, x1, y1 = active_region(rec.frame_size, pyautogui.size(), cfg.cursor_speed,
                                   cfg.area_center_y, cfg.area_height, cfg.area_center_x)
    region = (int(x0), int(y0), int(x1), int(y1))
    t0 = time.perf_counter()
    for i, (label, duration, text) in enumerate(steps):
        for phase, seconds in (("ready", READY_S), ("record", duration)):
            start = time.perf_counter()
            while (elapsed := time.perf_counter() - start) < seconds:
                frame = cam.read()
                if frame is None:
                    raise RuntimeError("Camera stopped sending frames")
                hands = tracker.process(frame)
                hand = hands[0] if hands else None

                rec.times.append(time.perf_counter() - t0)
                rec.labels.append(label if phase == "record" else "transition")
                rec.present.append(hand is not None)
                rec.image_lm.append(hand.landmarks if hand is not None else blank)
                rec.world_lm.append(hand.world if hand is not None and hand.world is not None
                                    else blank)

                draw_active_region(frame, region, hand is not None)
                if hand is not None:
                    draw_hand(frame, hand)
                step = f"Step {i + 1}/{len(steps)}"
                if phase == "ready":
                    draw_prompt(frame, f"{step}: get ready... {seconds - elapsed:.0f}",
                                text, elapsed / seconds, (80, 200, 255))
                else:
                    draw_prompt(frame, f"{step}: GO  ({seconds - elapsed:.0f}s)",
                                text, elapsed / seconds, (120, 230, 120))
                if hand is None:
                    cv2.putText(frame, "No hand detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (80, 80, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW,
                                                                   cv2.WND_PROP_VISIBLE) < 1:
                    return False
    return True


def main() -> None:
    wanted = sys.argv[1:]
    unknown = set(wanted) - {label for label, _, _ in STEPS}
    if unknown:
        print(f"Unknown steps: {', '.join(sorted(unknown))}. "
              f"Choose from: {', '.join(label for label, _, _ in STEPS)}")
        return
    steps = [st for st in STEPS if not wanted or st[0] in wanted]
    cfg = load_config()
    try:
        cam = Camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    except RuntimeError as exc:
        print(f"{exc}. Is AirMouse still running? Quit it (Ctrl+Alt+Q) and try again.")
        return

    rec = Recording(frame_size=cam.size)
    try:
        with cam, HandTracker(cfg) as tracker:
            cv2.namedWindow(WINDOW)
            rec.aborted = not run_steps(cam, tracker, cfg, rec, steps)
    finally:
        cv2.destroyAllWindows()
        rec.save()  # always: finished, stopped early, window closed, or crashed


if __name__ == "__main__":
    main()
