# AirMouse Roadmap

## Backend
- [x] **B0 Setup**: venv, dependencies, hand model, project rules
- [x] **B1 Vision core**: webcam capture + mirroring, 21-landmark tracking, skeleton overlay, FPS HUD
- [x] **B2 Cursor control**: active-region rectangle, map to screen resolution, EMA smoothing, move cursor (failsafe on)
- [x] **B3 Gesture engine**: hand-size-normalized pinch distances; left click (4↔8), right click (4↔12),
      scroll mode (index + middle up), hysteresis + cooldown debouncer, pinch-hold drag
- [x] **B4 Runtime + IPC**: engine on a worker thread, pause/resume, JSON config persistence,
      local WebSocket streaming video frames + state, accepting live config changes
- [x] **B5 Hardening**: pytest for the pure math (mapping, EMA, debouncer, classifier), code review, perf pass

## Open issues
- Pinch double click and joystick scroll: user reports some fixes still needed (to do).
- AirMouse.exe rebuilds are blocked by Smart App Control (unsigned); shortcuts use the signed
  pythonw + AirMouse.pyw launcher. Shipping the .exe needs a code-signing certificate.
- Thumb+index pinch (double click) and joystick scroll are tuned on synthetic data and checked
  for false triggers on real recordings; a `python -m airmouse.record pinch scroll` recording
  would confirm them on real pinches/scrolls.

## Frontend
- [x] **F1 Design**: "cyanotype" visual system; `frontend/` mockup replays a real recorded
      hand (DemoSource).
- [x] **F2 Shell**: tray app (pywebview window + pystray icon), live cyanotype camera, close
      hides to tray, single instance, streaming paused while hidden
- [x] **F3 Controls**: live sliders (two-way sync), on/off switch, "Fit area to my reach"
      calibration, Start with Windows (HKCU Run), sliding control area fitted to real reach
- [x] **F4 Ship**: review fixes (startup-error dialog + log, cleanup), bundled font, packaged
      `dist\AirMouse\AirMouse.exe` (PyInstaller, windowed), app icon, Desktop + Start menu shortcuts
