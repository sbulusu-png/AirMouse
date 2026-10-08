"""User-tunable settings: schema (for UI sliders), validation, and JSON persistence."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import USER_DIR, Config

CONFIG_PATH = USER_DIR / "settings.json"


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    group: str
    min: float
    max: float
    step: float
    integer: bool = False
    restart: str | None = None   # "camera" / "tracker": changing it rebuilds that component
    help: str = ""


PARAMS: tuple[Param, ...] = (
    Param("smoothing_alpha", "Smoothing", "Cursor", 0.05, 1.0, 0.01,
          help="Lower = steadier but laggier cursor"),
    Param("fingertip_follow", "Fingertip follow", "Cursor", 0.0, 1.0, 0.05,
          help="0 = cursor follows your hand and stays still while fingers gesture; "
               "1 = follows the fingertip exactly"),
    Param("cursor_speed", "Cursor speed", "Cursor", 1.0, 5.0, 0.1,
          help="Higher = less hand movement to cross the screen"),
    Param("taskbar_zone", "Taskbar zone", "Cursor", 0.0, 0.3, 0.01,
          help="Extra hand room for the taskbar: bigger = easier to reach and stay on (0 = off)"),
    Param("area_height", "Control area height", "Cursor", 0.25, 0.8, 0.01,
          help="Smaller = less up-and-down hand movement to reach the top and bottom"),
    Param("area_center_x", "Control area side", "Cursor", 0.25, 0.75, 0.01,
          help="Where your hand works left to right in the camera view"),
    Param("area_center_y", "Control area position", "Cursor", 0.25, 0.75, 0.01,
          help="Where your hand works in the camera view; lower = hand can sit lower"),
    Param("push_threshold", "Click sensitivity", "Gestures", 0.1, 0.8, 0.01,
          help="How far a finger must push forward to click; lower = easier"),
    Param("double_click_window", "Double-click time", "Gestures", 0.5, 2.5, 0.1,
          help="Two clicks on the same spot within this many seconds become a double click"),
    Param("click_cooldown", "Click cooldown", "Gestures", 0.1, 1.0, 0.05,
          help="Minimum seconds between clicks"),
    Param("scroll_speed", "Scroll speed", "Gestures", 200, 5000, 50, integer=True,
          help="How fast pages scroll while your hand is held above or below the scroll line"),
    Param("drag_deadzone", "Drag deadzone", "Gestures", 0, 150, 5, integer=True,
          help="Screen px of motion while pressed before a drag starts"),
    Param("click_lookback", "Click aim lookback", "Gestures", 0.0, 0.4, 0.01,
          help="On a click, snap the cursor to where it was this many seconds earlier"),
    Param("release_hold", "Release hold", "Gestures", 0.0, 0.5, 0.01,
          help="Keep the cursor pinned this long after a click"),
    Param("detection_confidence", "Detection confidence", "Tracking", 0.3, 0.95, 0.05,
          restart="tracker", help="Raise to reject false hands in low light"),
    Param("tracking_confidence", "Tracking confidence", "Tracking", 0.3, 0.95, 0.05,
          restart="tracker"),
    Param("camera_index", "Camera", "Tracking", 0, 9, 1, integer=True, restart="camera"),
)
PARAMS_BY_NAME = {p.name: p for p in PARAMS}


def schema() -> list[dict[str, Any]]:
    return [p.__dict__.copy() for p in PARAMS]


def to_dict(cfg: Config) -> dict[str, Any]:
    return {p.name: getattr(cfg, p.name) for p in PARAMS}


def apply_values(cfg: Config, values: dict[str, Any]) -> dict[str, Any]:
    """Validate, clamp and apply `values` to `cfg`. Returns the settings that changed.

    Raises ValueError on unknown names or non-numeric values (nothing is applied then).
    """
    clean: dict[str, Any] = {}
    for name, raw in values.items():
        param = PARAMS_BY_NAME.get(name)
        if param is None:
            raise ValueError(f"Unknown setting: {name}")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                or not math.isfinite(raw):  # json.loads accepts NaN / Infinity
            raise ValueError(f"{name} must be a finite number")
        value = min(max(float(raw), param.min), param.max)
        clean[name] = int(round(value)) if param.integer else value

    changed = {k: v for k, v in clean.items() if getattr(cfg, k) != v}
    for name, value in changed.items():
        setattr(cfg, name, value)
    return changed


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Defaults overlaid with saved settings. A corrupt file is set aside, not fatal."""
    cfg = Config()
    if not path.exists():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings file is not a JSON object")
        apply_values(cfg, {k: v for k, v in data.items() if k in PARAMS_BY_NAME})
    except (ValueError, OSError) as exc:
        backup = path.with_suffix(".bad.json")
        print(f"Ignoring unreadable settings ({exc}); moved to {backup}")
        path.replace(backup)
        cfg = Config()
    return cfg


def save_config(cfg: Config, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: never leaves a half-written settings file
