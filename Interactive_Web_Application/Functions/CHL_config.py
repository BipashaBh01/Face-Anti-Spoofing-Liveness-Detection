# ============================================================================
#  config.py — text and settings
# ============================================================================
#  Holds tunable values in one place. No markup or styling here; the page
#  is in the frontend folder.
# ============================================================================

from __future__ import annotations

from pathlib import Path

# Project root, derived from this file's location rather than the working
# directory, so paths hold regardless of where the app is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder holding the page, stylesheet and browser script.
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# Heading shown at the top of the page.
APP_TITLE = "Liveness Detection"

# Path to the downloaded MediaPipe face landmarker model.
LANDMARKER_MODEL = PROJECT_ROOT / "face_landmarker.task"

# The MediaPipe hand landmarker model, used for the hand challenge gestures.
HAND_LANDMARKER_MODEL = PROJECT_ROOT / "hand_landmarker.task"

# Folder where recordings are stored.
CAPTURE_DIR = PROJECT_ROOT / "output_challengeResponse" / "temp" / "liveness_captures"

# Folder for the challenge-response recordings, kept separate from the FAS
# recordings so the two are easy to tell apart.
CHALLENGE_CAPTURE_DIR = PROJECT_ROOT / "output_challengeResponse" / "temp" / "challenge_captures"
 
# Keep recordings after they are used.
# True  = keep the files (useful while building and testing).
# False = later, this will trigger auto-delete for production.
KEEP_RECORDINGS = True
 
# Frames per second used when assembling the recorded video.
CAPTURE_FPS = 8.0

# The study log for volunteer sessions.
SESSION_LOG_PATH = Path(__file__).resolve().parent.parent / "volunteer_session_data.xlsx"