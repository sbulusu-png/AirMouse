"""AirMouse entry point.

  python -m airmouse              tray app with the window (see app.py)
  python -m airmouse --minimized  start hidden in the tray (e.g. at login)
  python -m airmouse --preview    developer view: OpenCV window with the raw overlay

Gestures: point with the index finger to move. Left click: touch all five fingertips
together and open again, or tap the index finger forward and back; do either twice for a
double click. Right click: fold the thumb into the palm and back out. Drag: close the hand into a fist, move,
open to drop. Scroll: index+middle up together, move the hand up/down.

Keeps running when the window is minimized or another app has focus.
Global hotkeys (work from any app): Ctrl+Alt+M toggles cursor control, Ctrl+Alt+Q quits.
In the preview window: Space toggles, Q / Esc quits, and live tuning (saved):
  + / -  cursor speed      W / S  move the control area up / down
  ] / [  click sensitivity up / down (how easily a finger push clicks)
Failsafe: move the physical mouse into any screen corner to pause hand control.

Options: --port N, --paused (start with hand control off), --headless (with --preview:
no window at all).
"""

import argparse

from .instance import DEFAULT_PORT, show_running_instance


def main() -> None:
    parser = argparse.ArgumentParser(prog="airmouse")
    parser.add_argument("--preview", action="store_true",
                        help="developer view (OpenCV window) instead of the tray app")
    parser.add_argument("--minimized", action="store_true", help="start hidden in the tray")
    parser.add_argument("--headless", action="store_true", help="with --preview: no window")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--paused", action="store_true", help="start with hand control off")
    args = parser.parse_args()

    if args.preview or args.headless:
        from . import preview
        preview.run(args)
        return

    # Already running? Bring its window up and leave, before loading anything heavy.
    if not args.minimized and show_running_instance(args.port):
        print("AirMouse is already running; brought its window up.")
        return

    from .app import TrayApp, report_fatal
    try:
        TrayApp(args.port, paused=args.paused, minimized=args.minimized).run()
    except Exception as exc:
        report_fatal(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
