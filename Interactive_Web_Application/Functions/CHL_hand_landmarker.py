# ============================================================================
#  CHL_hand_landmarker.py — hand detection and finger states
# ============================================================================
#  Wraps the MediaPipe hand landmarker. For each frame it reports whether a
#  hand was found, which fingers are extended, and the raw landmarks, so the
#  hand challenge gestures can be recognised later from simple finger states.
# ============================================================================

from __future__ import annotations

# dataclass gives a small tidy result record.
from dataclasses import dataclass, field

# OpenCV converts the frame colour order for MediaPipe.
import cv2

# NumPy handles the landmark coordinates.
import numpy as np

# MediaPipe and its tasks API provide the hand landmarker.
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================================
#  RESULT RECORD
# ============================================================================

@dataclass
class HandResult:
    """What the hand landmarker found on one frame."""

    # Whether a hand was detected at all.
    found: bool = False

    # Which fingers are extended, in order:
    # [thumb, index, middle, ring, pinky]. True means extended.
    fingers: list = field(default_factory=lambda: [False, False, False, False, False])

    # How many fingers are extended in total.
    count: int = 0

    # The 21 hand landmarks as (x, y) pairs in the 0..1 range, or None.
    landmarks: list | None = None

    # Which hand it is, "Left" or "Right", as MediaPipe labels it, or "".
    handedness: str = ""

    # Whether the palm faces the camera roughly flat. When False the hand is
    # edge-on or turned away, and its finger states are not trustworthy.
    palm_faces_camera: bool = False


# ============================================================================
#  THE HAND LANDMARKER
# ============================================================================

class HandLandmarker:
    """Detects a hand and works out which fingers are extended."""

    def __init__(self, model_path) -> None:
        """Create the landmarker from the model file."""

        # Point the tasks API at the model file.
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

        # Configure for single images, one hand is enough for the gestures.
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
        )

        # Build the landmarker.
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def check(self, frame_bgr) -> HandResult:
        """Detect a hand on one BGR frame and report its finger states."""

        # MediaPipe expects RGB, so convert from OpenCV's BGR order.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wrap the frame for the tasks API.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection.
        result = self._landmarker.detect(mp_image)

        # No hand found: return the empty result.
        if not result.hand_landmarks:
            return HandResult(found=False)

        # Take the first hand's landmarks as (x, y) pairs for finger logic.
        points = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]

        # Also keep the (x, y, z) triples for the palm-facing check.
        points_3d = [(lm.x, lm.y, lm.z) for lm in result.hand_landmarks[0]]

        # Read the hand label (Left or Right) when present.
        handed = ""
        if result.handedness and result.handedness[0]:
            handed = result.handedness[0][0].category_name

        # Work out which fingers are extended.
        fingers = self._finger_states(points, handed)

        # Work out whether the palm faces the camera roughly flat.
        faces = self._palm_faces_camera(points_3d, handed)

        # Build and return the result.
        return HandResult(
            found=True,
            fingers=fingers,
            count=sum(1 for f in fingers if f),
            landmarks=points,
            handedness=handed,
            palm_faces_camera=faces,
        )

    def _finger_states(self, points, handed) -> list:
        """Return [thumb, index, middle, ring, pinky] extended flags.

        The landmark numbers follow MediaPipe's hand model. The four fingers
        are extended when the tip sits higher than the joint below it. The
        thumb is judged by distance from the palm, so it counts as extended
        whichever way it points (up, sideways, or out).
        """

        # Turn the points into an array for easy indexing.
        pts = np.array(points)

        # Landmark indices for each finger tip and the joint below it.
        # (tip, lower_joint) for index, middle, ring, pinky.
        finger_joints = [
            (8, 6),    # index
            (12, 10),  # middle
            (16, 14),  # ring
            (20, 18),  # pinky
        ]

        # Start with all fingers folded.
        states = [False, False, False, False, False]

        # ---- thumb, by distance from the palm ----
        # The palm centre is the average of the wrist and the four finger
        # bases. The thumb is extended when its tip is clearly farther from
        # the palm than its own base joint, which holds for any thumb
        # direction, not just sideways.
        palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)
        tip_dist = np.linalg.norm(pts[4] - palm)
        base_dist = np.linalg.norm(pts[2] - palm)
        states[0] = tip_dist > base_dist * 1.3

        # ---- other four fingers ----
        # A finger is extended when its tip sits higher (smaller y) than the
        # joint below it, since the image y grows downward.
        for slot, (tip, lower) in enumerate(finger_joints, start=1):
            states[slot] = pts[tip][1] < pts[lower][1]

        # Hand back the five flags.
        return states

    def thumb_pointing_up(self, points) -> bool:
        """Return True when the thumb points clearly upward.

        Used to tell a real "thumbs up" from a thumb merely sticking out
        sideways. The thumb tip must sit above its two joints below it.
        """

        # Turn the points into an array.
        pts = np.array(points)

        # Tip (4) higher than joint (3) higher than joint (2), in image y.
        return pts[4][1] < pts[3][1] < pts[2][1]

    def _palm_faces_camera(self, points_3d, handed) -> bool:
        """Return True when the palm (not the back of the hand) faces the
        camera roughly flat.

        The palm normal comes from two palm edge vectors. Its z part shows
        how much the hand faces the camera (magnitude) and which side faces
        it (sign). The sign that means palm-toward-camera is opposite for a
        left and a right hand, so handedness is used to pick it.
        """

        # Turn the triples into an array.
        pts = np.array(points_3d)

        # Three palm points: wrist (0), index base (5), pinky base (17).
        wrist = pts[0]
        index_base = pts[5]
        pinky_base = pts[17]

        # Two vectors across the palm.
        v1 = index_base - wrist
        v2 = pinky_base - wrist

        # The normal is their cross product, then made unit length.
        normal = np.cross(v1, v2)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            return False
        normal = normal / length

        # The z part: sign is front vs back, magnitude is flat vs edge-on.
        nz = normal[2]

        # DEBUG: print the signed z so the correct sign can be confirmed.
        # print(f"palm nz={nz:+.2f} hand={handed}")

        # Must be clearly flat-on, not edge-on.
        if abs(nz) < 0.5:
            return False

        # Palm toward the camera gives one sign; the back gives the other.
        # The sign flips between a left and a right hand.
        if handed == "Right":
            return nz < 0
        else:
            return nz > 0
    
