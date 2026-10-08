# AirMouse

Control your Windows mouse with bare-hand gestures in front of a webcam. AirMouse tracks
the 21 joints of your hand, draws them over the camera view, and turns gestures into
cursor movement, clicks, drags and scrolling. It runs quietly in the system tray.

## Gestures

| Gesture | Action |
|---|---|
| Point with your hand | Move the cursor |
| Touch all five fingertips together, then open | Click |
| Touch your thumb and index fingertips together | Double-click |
| Tap your index finger toward the screen | Quick click |
| Fold your thumb into your palm, then back out | Right-click |
| Close your fist, move, open your hand | Drag and drop |
| Two fingers up, then hold above or below where you started | Scroll |

The app window shows the live camera with your hand drawn on it, a guide that lights up
the gesture you're making, and settings. **Fit area to my reach** measures how far your
hand comfortably moves and sizes the control area to match.

**Hotkeys** (work from any app): `Ctrl+Alt+M` turns hand control on or off,
`Ctrl+Alt+Q` quits. **Emergency stop:** move the physical mouse into any screen corner.

## Requirements

- Windows 10 or 11 with a webcam
- Python 3.12
- Microsoft Edge WebView2 Runtime (included with Windows 11)

## Setup

```powershell
git clone https://github.com/sbulusu-png/AirMouse.git
cd AirMouse
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Hand-tracking model from Google MediaPipe (~8 MB, not stored in this repo)
mkdir models
curl.exe -L -o models\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run

```powershell
.venv\Scripts\python.exe -m airmouse
```

Or double-click `AirMouse.pyw` to start it without a console window. AirMouse lives in the
tray: click the tray icon to open the window; closing the window keeps it running. Turn on
**Start with Windows** in the window to have it start, hidden, when you sign in.

Options: `--minimized` (start in the tray), `--paused` (start with hand control off),
`--preview` (developer view with the raw tracking overlay).

## Privacy

Everything runs on your computer. Camera images are never saved or sent anywhere. The
window talks to the tracking engine over a server bound to `127.0.0.1` that requires a
random per-session token, so web pages can't connect to it. Settings are stored in
`%APPDATA%\AirMouse`.

## Development

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Tests need no camera: they drive the gesture engine with synthetic hands. Gesture
thresholds were tuned on recordings of a real hand; to record your own, run
`.venv\Scripts\python.exe -m airmouse.record` (joint coordinates only, no video).

### Building an .exe

```powershell
.venv\Scripts\python.exe -m PyInstaller packaging\airmouse.spec --noconfirm
```

This produces `dist\AirMouse\AirMouse.exe`. Windows **Smart App Control** blocks unsigned
programs it doesn't recognise, so on PCs where it's on the build needs a code-signing
certificate; running from source (above) works without one.

## How it works

- `camera.py`: webcam capture on its own thread, mirrored frames
- `tracker.py`: MediaPipe Tasks hand landmarker
- `gestures.py`: gesture recognition from 3D joint geometry
- `cursor.py`: control-area mapping, smoothing, click aim pinning, mouse output
- `engine.py`: the tracking loop on a background thread
- `server.py`: local WebSocket and HTTP bridge to the window
- `app.py`: tray icon and window; `frontend/`: the window's UI

## License

MIT, see [LICENSE](LICENSE). Bundled third-party material: Schibsted Grotesk font, SIL Open
Font License 1.1 ([frontend/fonts/OFL.txt](frontend/fonts/OFL.txt)). The hand-tracking model
is downloaded from Google MediaPipe and is not part of this repository.
