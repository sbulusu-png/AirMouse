"""Developer view: OpenCV window with the raw overlay (python -m airmouse --preview).

Keys: Space toggles hand control, Q / Esc quits, and live tuning (saved):
  + / -  cursor speed      W / S  move the control area up / down
  ] / [  click sensitivity up / down (how easily a finger push clicks)
"""

import argparse
import time

import cv2

from .engine import Engine
from .hotkeys import QUIT_LABEL, TOGGLE_LABEL, Hotkeys
from .overlay import draw_active_region, draw_hand, draw_hud, draw_pointer
from .server import Server
from .settings import CONFIG_PATH, load_config, save_config
from .tracker import INDEX_TIP

WINDOW = "AirMouse"

# Preview-window key -> (setting, step). Clamped to the setting's range and saved.
TUNING_KEYS = {
    ord("+"): ("cursor_speed", 0.2), ord("="): ("cursor_speed", 0.2),
    ord("-"): ("cursor_speed", -0.2), ord("_"): ("cursor_speed", -0.2),
    ord("w"): ("area_center_y", -0.03), ord("s"): ("area_center_y", 0.03),
    ord("]"): ("push_threshold", -0.03), ord("["): ("push_threshold", 0.03),
}


def tune(engine: Engine, name: str, step: float) -> None:
    current = engine.config_dict()[name]
    engine.update_config({name: round(current + step, 3)})
    save_config(engine.cfg)
    print(f"{name} = {engine.config_dict()[name]}")


def show_preview(engine: Engine) -> int:
    """Draw the latest frame with overlays; returns the pressed key (or -1)."""
    snap = engine.snapshot()
    if snap.frame is None:
        return cv2.waitKey(30)
    frame = snap.frame.copy()
    s = snap.state
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = s["active_region"]
    draw_active_region(frame, (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)),
                       s["enabled"] and snap.hand is not None)
    if snap.hand is not None:
        draw_hand(frame, snap.hand)
        draw_pointer(frame, snap.hand, INDEX_TIP)
    hud = [f"FPS {s['fps']:4.1f}",
           f"{snap.hand.handedness} hand ({snap.hand.score:.2f})" if snap.hand else "No hand",
           f"Cursor control {'ON' if s['enabled'] else 'OFF'}  [{TOGGLE_LABEL}]"]
    if s["enabled"] and snap.hand is not None:
        hud.append(f"{s['mode']}   index push {s['left_push']:+.2f}"
                   f"  middle {s['right_push']:+.2f}"
                   f"  (click > {engine.cfg.push_threshold:.2f})")
    cfg = engine.cfg
    hud.append(f"Speed {cfg.cursor_speed:.1f} [-/+]   Area [W/S]   "
               f"Click sensitivity {cfg.push_threshold:.2f} [ [ / ] ]")
    draw_hud(frame, hud)
    cv2.imshow(WINDOW, frame)
    return cv2.waitKey(15)


def run(args: argparse.Namespace) -> None:
    engine = Engine(load_config(), enabled=not args.paused)
    server = Server(engine, port=args.port)
    engine.on_update = server.publish
    server.start()
    engine.start()
    print(f"AirMouse running. {TOGGLE_LABEL}: toggle hand control, {QUIT_LABEL}: quit.")
    print(f"Settings: {CONFIG_PATH}")
    print(f"UI endpoint: {server.url}")

    preview = not args.headless
    if preview:
        # Create the window now: until the first imshow it doesn't exist, and a missing
        # window reads as "closed by the user" below, which would quit immediately.
        cv2.namedWindow(WINDOW)
    try:
        with Hotkeys() as hotkeys:
            while engine.alive and not hotkeys.quit_requested:
                key = (show_preview(engine) & 0xFF) if preview else -1
                if not preview:
                    time.sleep(0.05)
                if key == ord(" ") or hotkeys.take_toggle():
                    engine.set_enabled(not engine.enabled)
                    print(f"Hand control {'ON' if engine.enabled else 'OFF'}")
                elif key in TUNING_KEYS:
                    tune(engine, *TUNING_KEYS[key])
                elif key in (ord("q"), 27):
                    break
                elif preview and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.stop()
        cv2.destroyAllWindows()
