"""Tray app startup paths that don't need a window, camera or tray."""

import ctypes
import socket

import pytest

import airmouse.app as app_mod
import airmouse.config as config_mod
from airmouse.app import TrayApp, report_fatal, show_running_instance


@pytest.fixture
def busy_port():
    """A port held by some other program (not AirMouse)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    yield s.getsockname()[1]
    s.close()


def test_other_program_on_port_is_not_mistaken_for_airmouse(busy_port):
    assert show_running_instance(busy_port) is False


def test_port_taken_by_another_program_fails_clearly(busy_port, monkeypatch):
    started = []
    monkeypatch.setattr(app_mod.Engine, "start", lambda self: started.append(1))
    with pytest.raises(RuntimeError, match="another program may be using it"):
        TrayApp(busy_port, paused=True, minimized=True).run()
    assert started == []                              # the camera engine never started


def test_fatal_error_is_logged_and_shown(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(config_mod, "USER_DIR", tmp_path)
    monkeypatch.setattr(ctypes.windll.user32, "MessageBoxW",
                        lambda hwnd, text, title, flags: shown.append((text, title)))
    try:
        raise RuntimeError("camera exploded")
    except RuntimeError as exc:
        report_fatal(exc)
    log = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "camera exploded" in log and "Traceback" in log
    text, title = shown[0]
    assert "camera exploded" in text and str(tmp_path / "error.log") in text
    assert title == "AirMouse couldn't start"
