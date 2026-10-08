"""AirMouse as a tray app: it runs in the background and opens its window on demand,
like a VPN client.

* Tray icon (by the clock): left-click opens the window; right-click for Open / Hand
  control / Quit. The icon greys out while hand control is off.
* Closing the window only hides it; hand control keeps running. Quit from the tray menu
  or with Ctrl+Alt+Q.
* Only one copy runs: launching AirMouse again just brings up the running one's window.
* While the window is hidden, no video or state is streamed, saving CPU.
"""

import json
import threading
from typing import Any

import pystray
import webview
from PIL import Image, ImageDraw

from .config import PROJECT_ROOT
from .engine import Engine
from .hotkeys import QUIT_LABEL, TOGGLE_LABEL, Hotkeys
from .instance import show_running_instance
from .server import Server
from .settings import load_config
from .tracker import HAND_CONNECTIONS

APP_NAME = "AirMouse"
GLYPHS = PROJECT_ROOT / "frontend" / "glyphs.json"
WINDOW_SIZE = (1180, 760)
MIN_SIZE = (820, 560)


def tray_image(enabled: bool, size: int = 64) -> Image.Image:
    """The pointing-hand skeleton drawn in light on cyanotype blue (grey when paused)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((1, 1, size - 2, size - 2), radius=size // 4,
                           fill="#1f4e8c" if enabled else "#7a8693")
    pts = json.loads(GLYPHS.read_text(encoding="utf-8"))["move"]
    xy = [(size / 2 + x * size * 0.72, size / 2 + y * size * 0.72) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        draw.line([xy[a], xy[b]], fill="#eaf2f8", width=max(2, size // 20))
    return img


class TrayApp:
    def __init__(self, port: int, paused: bool, minimized: bool) -> None:
        self.port = port
        self.paused = paused
        self.minimized = minimized
        self._quitting = False
        self._shown_enabled: bool | None = None

    def run(self) -> None:
        if show_running_instance(self.port):
            print("AirMouse is already running; brought its window up.")
            return

        self.engine = Engine(load_config(), enabled=not self.paused)
        self.server = Server(self.engine, port=self.port)
        self.engine.on_update = self.server.publish
        self.server.on_show = self.show
        try:
            self.server.start()
        except RuntimeError as exc:
            raise RuntimeError(f"AirMouse couldn't open its local port {self.port}; another "
                               f"program may be using it.\n\n{exc}") from exc
        self.engine.start()
        try:
            self._run_ui()
        finally:
            self._quitting = True
            self.engine.stop()
            self.server.stop()

    def _run_ui(self) -> None:
        self.server.set_streaming(not self.minimized)

        window = webview.create_window(
            APP_NAME, self.server.http_url, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1],
            min_size=MIN_SIZE, hidden=self.minimized, background_color="#F2F5F7")
        if window is None:
            raise RuntimeError("Could not create the AirMouse window")
        self.window = window
        self.window.events.closing += self._on_closing

        self.icon = pystray.Icon(APP_NAME, tray_image(self.engine.enabled), APP_NAME,
                                 menu=pystray.Menu(
            pystray.MenuItem("Open AirMouse", self._on_open, default=True),
            pystray.MenuItem(f"Hand control ({TOGGLE_LABEL})", self._on_toggle,
                             checked=lambda item: self.engine.enabled),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(f"Quit AirMouse ({QUIT_LABEL})", self._on_quit),
        ))
        self.icon.run_detached()
        threading.Thread(target=self._watch, name="airmouse-tray", daemon=True).start()
        print(f"AirMouse running in the tray. {TOGGLE_LABEL}: hand control, {QUIT_LABEL}: quit.")
        print(f"UI: {self.server.http_url}")

        try:
            webview.start(private_mode=True)   # returns once the window is destroyed (Quit)
        finally:
            self._quitting = True
            self.icon.stop()

    # ---- window ------------------------------------------------------------------------

    def show(self) -> None:
        self.server.set_streaming(True)
        self.window.show()
        self.window.restore()

    def hide(self) -> None:
        self.window.hide()
        self.server.set_streaming(False)

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self.window.destroy()

    def _on_closing(self) -> bool:
        if self._quitting:
            return True
        # Close = hide to the tray. Hide from another thread: this runs on the UI thread.
        threading.Thread(target=self.hide, daemon=True).start()
        return False

    # ---- tray --------------------------------------------------------------------------

    def _on_open(self, icon: Any, item: Any) -> None:
        self.show()

    def _on_toggle(self, icon: Any, item: Any) -> None:
        self.engine.set_enabled(not self.engine.enabled)

    def _on_quit(self, icon: Any, item: Any) -> None:
        self.quit()

    def _watch(self) -> None:
        """Global hotkeys, and keep the tray icon in step with hand control (which can
        also change from the window, the hotkey or the failsafe)."""
        with Hotkeys() as hotkeys:
            while not self._quitting:
                if hotkeys.take_toggle():
                    self.engine.set_enabled(not self.engine.enabled)
                if hotkeys.quit_requested:
                    self.quit()
                    return
                enabled = self.engine.enabled
                if enabled != self._shown_enabled:
                    self._shown_enabled = enabled
                    self.icon.icon = tray_image(enabled)
                    self.icon.title = f"{APP_NAME}: hand control {'on' if enabled else 'off'}"
                    self.icon.update_menu()
                threading.Event().wait(0.1)


def report_fatal(exc: BaseException) -> None:
    """Without a console (AirMouse.exe / AirMouse.pyw) a crash would be invisible: write the
    details to a log and tell the user in a dialog."""
    import ctypes
    import traceback

    from .config import USER_DIR

    log = USER_DIR / "error.log"
    try:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    except OSError:
        pass
    MB_ICONERROR = 0x10
    ctypes.windll.user32.MessageBoxW(
        None, f"{exc}\n\nDetails were saved to {log}", "AirMouse couldn't start", MB_ICONERROR)
