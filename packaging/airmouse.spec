# PyInstaller build for AirMouse.exe (a windowed tray app, no console).
#
#   .venv\Scripts\python.exe -m PyInstaller packaging\airmouse.spec --noconfirm
#
# Output: dist\AirMouse\AirMouse.exe (one folder: starts faster than a single-file build
# and is friendlier to antivirus scanners). Settings and logs live in %APPDATA%\AirMouse.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent

# MediaPipe has no PyInstaller hook: take its data files and native libmediapipe.dll.
mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")

a = Analysis(
    [str(ROOT / "AirMouse.pyw")],
    pathex=[str(ROOT)],
    binaries=mp_binaries,
    datas=mp_datas + [
        (str(ROOT / "frontend"), "frontend"),
        (str(ROOT / "models" / "hand_landmarker.task"), "models"),
    ],
    hiddenimports=mp_hidden + [
        "airmouse.app",                     # imported lazily by __main__
        "pystray._win32",
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
    ],
    excludes=["pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AirMouse",
    icon=str(ROOT / "packaging" / "airmouse.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="AirMouse", upx=False)
