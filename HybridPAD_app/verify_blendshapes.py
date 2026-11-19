# ============================================================================
#  verify_blendshapes.py — see which blendshapes actually respond for you
# ============================================================================
#  Prints the live scores of several candidate blendshapes so we can pick
#  reliable ones for the face challenge. Try each expression and watch which
#  numbers actually move.
#
#  Run from the project root:  python verify_blendshapes.py
#  Ctrl+C to stop.
# ============================================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from Functions.CHL_config import LANDMARKER_MODEL

# Build a blendshape-enabled landmarker.
base = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_MODEL))
opts = vision.FaceLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True,
    num_faces=1,
)
lm = vision.FaceLandmarker.create_from_options(opts)

# The candidate blendshapes to watch (the potentially useful ones).
WATCH = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "mouthFunnel", "mouthPucker",
    "jawLeft", "jawRight", "mouthLeft", "mouthRight",
    "mouthFrownLeft", "mouthFrownRight", "cheekPuff",
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "noseSneerLeft", "noseSneerRight", "eyeSquintLeft", "eyeSquintRight",
    'mouthRollUpper','mouthRollLower',
]

print("Try each expression and watch which numbers move above ~0.4.")
print("Ctrl+C to stop.\n")

cap = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = lm.detect(img)
        if not res.face_blendshapes:
            print("no face")
            continue
        d = {b.category_name: b.score for b in res.face_blendshapes[0]}
        # Print only the ones currently above 0.30, to spot what responds.
        active = {k: round(d.get(k, 0), 2) for k in WATCH if d.get(k, 0) > 0.30}
        print(active if active else "-")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()