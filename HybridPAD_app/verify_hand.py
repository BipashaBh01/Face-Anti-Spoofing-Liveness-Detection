# ============================================================================
#  verify_hand.py — quick check that the hand landmarker works
# ============================================================================
#  Opens the webcam, runs the hand landmarker on each frame, and prints the
#  finger states to the terminal. Use it to confirm the model loads and the
#  finger detection is sensible before wiring it into the app.
#
#  Run from the project root:
#      python verify_hand.py
#  Press Ctrl+C to stop.
# ============================================================================

# OpenCV grabs webcam frames.
import cv2

# The hand landmarker wrapper.
from Functions.CHL_hand_landmarker import HandLandmarker

# The model path from config.
from Functions.CHL_config import HAND_LANDMARKER_MODEL


# Build the landmarker.
hand = HandLandmarker(HAND_LANDMARKER_MODEL)
print("Hand landmarker created. Show your hand to the camera.")
print("Finger order printed as [thumb, index, middle, ring, pinky].")

# Open the default webcam.
cap = cv2.VideoCapture(0)

# A short name for common finger patterns, to make the output readable.
def name_for(fingers, count):
    # Compare against a few well-known shapes.
    t, i, m, r, p = fingers
    if count == 0:
        return "fist"
    if count == 5:
        return "open palm (5)"
    if i and m and not r and not p and not t:
        return "peace / two"
    if i and not m and not r and not p:
        return "one finger"
    if i and m and r and not p:
        return "three"
    if i and m and r and p and not t:
        return "four"
    if t and not i and not m and not r and not p:
        return "thumb out (up or sideways)"
    if i and p and not m and not r:
        return "rock sign"
    return "other"

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

        # Print what was found.
        if res.found:
            # Check specifically whether the thumb points up.
            up = hand.thumb_pointing_up(res.landmarks)
            label = name_for(res.fingers, res.count)
            if res.fingers[0] and res.count == 1 and up:
                label = "thumbs UP"
            print(f"hand={res.handedness:<5} fingers={res.fingers} "
                  f"count={res.count} thumb_up={up}  -> {label}")
        else:
            print("no hand")

except KeyboardInterrupt:
    # Stop cleanly on Ctrl+C.
    print("\nStopped.")

finally:
    # Release the camera.
    cap.release()