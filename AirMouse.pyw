"""Double-click to start AirMouse without a console window (it lives in the system tray).

Also what Windows runs at sign-in when "Start with Windows" is on.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from airmouse.__main__ import main  # noqa: E402

main()
