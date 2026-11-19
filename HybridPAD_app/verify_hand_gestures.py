# ============================================================================
#  verify_hand_gestures.py — check the 8 hand gesture detectors
# ============================================================================
#  Opens the webcam, reads the hand finger states, and prints which of the
#  eight gestures fires. Use it to confirm each gesture is recognised before
#  wiring the detectors into the challenge validation.
#
#  Run from the project root:
#      python verify_hand_gestures.py
#  Show each gesture and watch which one fires. Ctrl+C to stop.
# ============================================================================

# OpenCV grabs webcam frames.
import cv2

# The hand landmarker wrapper.
from Functions.CHL_hand_landmarker import HandLandmarker

# The hand gesture detectors.
from Functions.CHL_primitives import HAND_DETECTORS

# The hand model path from config.
from Functions.CHL_config import HAND_LANDMARKER_MODEL


# Build the landmarker.
hand = HandLandmarker(HAND_LANDMARKER_MODEL)

print("Show each gesture: one finger, peace, three, four, five (open palm),")
print("fist, thumbs up, rock sign. The fired gesture prints each frame.")
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

        # Run the hand landmarker.
        res = hand.check(frame)

        # No hand: say so.
        if not res.found:
            print("no hand")
            continue

        # Work out whether the thumb points up, for thumbs up.
        thumb_up = hand.thumb_pointing_up(res.landmarks)

        # See which gestures fire, passing the palm-facing flag.
        fired = [key for key, fn in HAND_DETECTORS.items()
                 if fn(res.fingers, thumb_up, res.palm_faces_camera)]

        # Print the finger states, palm flag, and any fired gesture.
        print(f"fingers={res.fingers} thumb_up={thumb_up} "
              f"palm={res.palm_faces_camera}  -> {fired if fired else '-'}")

except KeyboardInterrupt:
    # Stop cleanly on Ctrl+C.
    print("\nStopped.")

finally:
    # Release the camera.
    cap.release()