# ============================================================================
#  verify_head.py — check the head detectors against real movement
# ============================================================================
#  Opens the webcam, reads the head angles with the existing position
#  checker, and prints which head actions fire on each frame. Use it to
#  confirm the angle signs and thresholds match real left/right/up/down/tilt
#  movements before wiring the detectors into the challenge validation.
#
#  Run from the project root:
#      python verify_head.py
#  Move your head each way and watch which action fires. Ctrl+C to stop.
# ============================================================================

# OpenCV grabs webcam frames.
import cv2

# The position checker gives yaw, pitch, roll per frame.
from Functions.CHL_position_checker import PositionChecker, PositionConfig
from Functions.CHL_config import LANDMARKER_MODEL

# The head detectors.
from Functions.CHL_primitives import HEAD_DETECTORS


# Build the position checker.
checker = PositionChecker(LANDMARKER_MODEL, PositionConfig())
print("Move your head: turn left/right, tilt left/right, look up/down.")
print("The fired action and the raw angles print each frame. Ctrl+C to stop.")

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

        # Run the position check to get the angles.
        result = checker.check(frame)

        # Read the three angles.
        yaw = result.yaw
        pitch = result.pitch
        roll = result.roll

        # See which head actions fire on this frame.
        fired = [key for key, fn in HEAD_DETECTORS.items()
                 if fn(yaw, pitch, roll)]

        # Print the angles and any fired action.
        print(f"yaw={yaw:6.1f} pitch={pitch:6.1f} roll={roll:6.1f}  "
              f"-> {fired if fired else '-'}")

except KeyboardInterrupt:
    # Stop cleanly on Ctrl+C.
    print("\nStopped.")

finally:
    # Release the camera.
    cap.release()