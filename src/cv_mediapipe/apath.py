import sys
from pathlib import Path

import cv_mediapipe


_pwd = Path(cv_mediapipe.__file__).parent
# For PyInstaller
pwd = Path(getattr(sys, "_MEIPASS", _pwd))
ASSETS_PATH = pwd / "assets" 
