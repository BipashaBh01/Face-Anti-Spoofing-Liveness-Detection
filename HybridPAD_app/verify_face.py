# ============================================================================
#  verify_face.py — check the face detectors and blink counter
# ============================================================================
#  Opens the webcam, reads the face blendshapes, prints the raw scores and
#  which face actions fire, and keeps a running blink count. Use it to
#  confirm the thresholds match real expressions before wiring in.
#
#  Run from the project root:
#      python verify_face.py
#  Try: open mouth, smile, puff cheeks, raise eyebrows, blink once, blink
#  twice. Watch the scores and fired actions. Ctrl+C to stop.
# ============================================================================

# OpenCV grabs webcam frames.
import cv2

# The face reader, detectors and blink counter.
from Functions.CHL_face_blendshapes import (
    FaceReader,
    face_open_mouth,
    face_smile,
    face_roll_lips,
    face_raise_brows,
    BlinkCounter,
)

# The face model path from config.
from Functions.CHL_config import LANDMARKER_MODEL


# Build the reader and the blink counter.
reader = FaceReader(LANDMARKER_MODEL)
blinks = BlinkCounter()

print("Try: open mouth, smile, puff cheeks, raise eyebrows, and blink.")
print("Scores, fired actions and the running blink count print each frame.")
print("Ctrl+C to stop.")

# Open the default webcam.
cap = cv2.VideoCapture(0)

try:
    # Loop over frames.
    while True:

        # Read one frame.
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        # Read the blendshape scores.
        s = reader.read(frame)

        # Update the blink count.
        blinks.update(s)

        # No face: say so and continue.
        if not s.found:
            print("no face")
            continue

        # See which simple actions fire.
        fired = []
        if face_open_mouth(s):
            fired.append("open_mouth")
        if face_smile(s):
            fired.append("smile")
        if face_roll_lips(s):
            fired.append("roll_lips")
        if face_raise_brows(s):
            fired.append("raise_brows")

        # Print the scores, fired actions and blink count.
        print(f"jaw={s.jaw_open:.2f} smile={s.smile:.2f} "
              f"roll={s.roll_lips:.2f} brow={s.brow_raise:.2f} "
              f"blink={s.eye_blink:.2f} | blinks={blinks.count} "
              f"| {fired if fired else '-'}")

except KeyboardInterrupt:
    # Stop cleanly on Ctrl+C.
    print("\nStopped.")

finally:
    # Release the camera.
    cap.release()