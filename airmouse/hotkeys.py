"""System-wide hotkeys, so AirMouse can be controlled while another app has focus."""

import threading

from pynput import keyboard

TOGGLE_HOTKEY = "<ctrl>+<alt>+m"
QUIT_HOTKEY = "<ctrl>+<alt>+q"
TOGGLE_LABEL = "Ctrl+Alt+M"
QUIT_LABEL = "Ctrl+Alt+Q"


class Hotkeys:
    """Listens on a background thread; the main loop polls the request flags."""

    def __init__(self) -> None:
        self._toggle = threading.Event()
        self._quit = threading.Event()
        self._listener = keyboard.GlobalHotKeys({
            TOGGLE_HOTKEY: self._toggle.set,
            QUIT_HOTKEY: self._quit.set,
        })

    def take_toggle(self) -> bool:
        """True once per toggle-hotkey press."""
        if self._toggle.is_set():
            self._toggle.clear()
            return True
        return False

    @property
    def quit_requested(self) -> bool:
        return self._quit.is_set()

    def __enter__(self) -> "Hotkeys":
        self._listener.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._listener.stop()
