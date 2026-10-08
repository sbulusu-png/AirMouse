# AirMouse

Control the Windows mouse with bare-hand gestures from a webcam.

## Stack
- Python 3.12, venv at `.venv` (`.venv\Scripts\python.exe`)
- MediaPipe **1.0.1**: the legacy `mp.solutions.hands` API is GONE. Use the Tasks API
  (`mediapipe.tasks.python.vision.HandLandmarker`, `RunningMode.VIDEO`, `detect_for_video`)
  with the model at `models/hand_landmarker.task`.
- OpenCV 5, NumPy 2, PyAutoGUI, pynput
- Frontend (later): HTML/CSS/JS in a pywebview window, fed by a local WebSocket from the backend.

## Run
`.venv\Scripts\python.exe -m airmouse`: tray app (`app.py`). Flags: `--minimized`, `--paused`,
`--port N`; `--preview` = developer OpenCV view. `AirMouse.pyw` = no-console launcher (also used by
Start with Windows, `autostart.py`). The server also serves `frontend/` over HTTP (see server.py).

Restart the app after code changes; a running copy keeps the old code.

## Build
`.venv\Scripts\python.exe -m PyInstaller packaging\airmouse.spec --noconfirm` ->
`dist\AirMouse\AirMouse.exe` (windowed, ~300 MB, MediaPipe collected via collect_all). The Desktop
and Start menu shortcuts point there, so rebuild after changes. Packaged, `config.FROZEN` is
True: resources come from `sys._MEIPASS`, logs go to `%APPDATA%\AirMouse\logs`, startup errors
show a dialog and write `%APPDATA%\AirMouse\error.log`. Close the running app before rebuilding.

**Smart App Control is ON on this PC** and blocks rebuilt (unsigned, unknown) AirMouse.exe
builds. Don't change that setting. The Desktop/Start-menu shortcuts and Start with Windows
therefore launch the signed `.venv\Scripts\pythonw.exe AirMouse.pyw` instead (no console,
same tray app). The .exe route needs a real code-signing certificate.

## Test
`.venv\Scripts\python.exe -m pytest -q` and `npx -y pyright` (both must stay clean).
Tests need no camera: `tests/hands.py` builds synthetic hands; `tests/test_engine.py` fakes
camera/tracker/mouse. Dev deps: `requirements-dev.txt`.

## Architecture
- `camera.py` capture thread (MSMF, 30 fps; DSHOW measured slower here) → mirrored frames
- `engine.py` worker thread: track → `gestures.py` → `cursor.py` (mouse); thread-safe API
- `server.py` WebSocket on 127.0.0.1:8765, per-run token in the URL; protocol in its docstring
- `settings.py` param schema/ranges (drives UI sliders) + JSON at `%APPDATA%\AirMouse\settings.json`
- Clicks are finger pushes toward the camera (foreshortening + MediaPipe z vs. a slow baseline)
- Fist = grab (left button held, cursor follows the index knuckle); drop when 2+ fingers
  open for 0.12 s, or one finger (pointing again) for 0.3 s.
- Right click = thumb folded into the palm (tip -> pinky knuckle < 0.65, from > 1.0 within
  0.4 s, other fingers' reach unchanged) and back out. A movement, not a pose: the thumb
  often rests tucked while pointing/scrolling. The middle finger no longer clicks.
- Open/curled/fist use 3D reach (wrist->tip incl. MediaPipe z / 3D palm), tuned on real
  logs; pushes must be isolated (beat the other fingers incl. ring) to click.
- Taps click on pull-back, only if fast (rest->pressed < 0.35 s), brief (< 1 s) and the
  hand stays put; a long push is a posture (pointing at the camera) and becomes the new
  rest. Taps never hold the button: dragging is fist-only.
- Five-finger touch (all tips within 0.45 palm of the thumb tip AND the tip cluster out in
  front of the palm centre, > 0.38 -- a fist tucks it in; 2 consecutive frames) =
  left click on release; overrides a push in progress. A second left click within
  `double_click_window` (1.5 s) and ~60 px is
  made into a real double click by the engine (same spot; doubleClick() if too slow).
- Cursor aims at the index knuckle lifted 0.78 palm lengths (where the straight finger
  points), not the raw tip (#8): gestures move the tip 1.7-3x more than the knuckle.
  `fingertip_follow` (0..1) blends back toward the tip.
- Control area slides with the hand past its edges (same speed); "Fit area to my reach"
  (engine.fit_area) sets speed/height/centre from an 8 s sweep.
- Double click = thumb+index pinch (tips < 0.25, middle/ring/pinky tips > 0.45 from the
  thumb, not a fist), fired on close; 0 false pinches on all recorded data. Two quick
  five-finger touches still pair into a double click as a fallback.
- Scroll is a joystick: the hand height when the 2-finger pose starts is the "scroll line";
  held above/below it scrolls up/down (speed ~ distance beyond a 0.12-palm dead zone).
  Stroke-following made return strokes cancel (3,521 up vs 4,021 down in 8 s).
- Taskbar zone: the hand area's edge slice (`taskbar_zone`, 0.15) maps onto the taskbar
  strip found from Windows' work area (cursor.taskbar_bands), so it's easier to reach/hold.
- Startup: `__main__` checks for a running copy (instance.py, stdlib only) BEFORE heavy
  imports, so a second launch shows the window in <1 s.
- Tuning workflow: `python -m airmouse.record` (guided, saves logs/recording-*.npz) or
  `session_log=True` in config; replay logs through GestureEngine to test changes.

## Rules
Follow `.claude/skills/gesture-rules` for all gesture code: mirror first, EMA smoothing,
`pyautogui.FAILSAFE = True`, map normalized 0–1 coords to screen pixels.

## Roadmap
See `ROADMAP.md`.
