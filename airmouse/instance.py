"""Single-instance hand-off, using only the standard library so it runs before any heavy
import: a second launch of AirMouse brings up the running one's window almost instantly
instead of first loading OpenCV and MediaPipe for several seconds."""

import urllib.request

DEFAULT_PORT = 8765
SHOW_HEADER = "X-AirMouse"


def show_running_instance(port: int = DEFAULT_PORT) -> bool:
    """Ask an already-running AirMouse to show its window. True if one answered."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/show",
                                 headers={SHOW_HEADER: "show"})
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False
