"""Start AirMouse when the user signs in to Windows (hidden in the tray, like a VPN client).

Uses the per-user Run key (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run): no admin
rights, and only touched when the user flips the switch in the window.
"""

import sys
import winreg
from pathlib import Path

from .config import FROZEN, PROJECT_ROOT

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "AirMouse"
LAUNCHER = PROJECT_ROOT / "AirMouse.pyw"


def command() -> str:
    """Start hidden in the tray: AirMouse.exe when packaged, otherwise pythonw (no console
    window) running the launcher script."""
    if FROZEN:
        return f'"{sys.executable}" --minimized'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return f'"{pythonw}" "{LAUNCHER}" --minimized'


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
        return value == command()
    except OSError:
        return False


def set_enabled(enabled: bool) -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        if enabled:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command())
        else:
            try:
                winreg.DeleteValue(key, VALUE_NAME)
            except FileNotFoundError:
                pass
