# ============================================================================
#  CHL_face_blendshapes.py — read face blendshapes and detect face actions
# ============================================================================
#  The position checker reads head angles, but the face challenge actions
#  need blendshapes (smile, open mouth, and so on). This module runs its own
#  face landmarker with blendshapes turned on, and offers per-frame detectors
#  for the simple face actions. Blink counting is stateful and handled by a
#  separate small counter class.
# ============================================================================

from __future__ import annotations

# dataclass gives a tidy per-frame record.
from dataclasses import dataclass

# OpenCV converts colour order for MediaPipe.
import cv2

# MediaPipe and its tasks API provide the face landmarker with blendshapes.
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================================
#  THRESHOLDS
# ============================================================================
#  Each blendshape score runs 0..1. These thresholds are set where a clear,
#  deliberate action stands out from a resting face, and can be tuned after
#  real testing.
# ============================================================================

# Mouth open enough to count as "open your mouth".
OPEN_MOUTH_THRESH = 0.40

# Smile strength (averaged over both sides) to count as a smile.
SMILE_THRESH = 0.40

# Lip-roll strength (averaged over upper and lower) to count as rolling the
# lips inward.
ROLL_LIPS_THRESH = 0.35

# Eyebrow raise strength (averaged) to count as raised eyebrows.
BROW_THRESH = 0.55

# Eye-blink strength above which an eye counts as closed, for blink counting.
BLINK_CLOSED_THRESH = 0.50


# ============================================================================
#  BLENDSHAPE READER
# ============================================================================

@dataclass
class FaceScores:
    """The blendshape scores needed for the face actions, for one frame."""

    # Whether a face was found.
    found: bool = False

    # The individual scores, each 0..1.
    jaw_open: float = 0.0
    smile: float = 0.0
    roll_lips: float = 0.0
    brow_raise: float = 0.0
    eye_blink: float = 0.0  # averaged over both eyes


class FaceReader:
    """Runs a face landmarker with blendshapes and returns the scores."""

    def __init__(self, model_path) -> None:
        """Create the landmarker from the model file."""

        # Point the tasks API at the model file.
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

        # Turn blendshapes on; one face is enough for the challenge.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )

        # Build the landmarker.
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def read(self, frame_bgr) -> FaceScores:
        """Read the needed blendshape scores from one BGR frame."""

        # MediaPipe expects RGB.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wrap the frame for the tasks API.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection.
        result = self._landmarker.detect(mp_image)

        # No face or no blendshapes: return the empty record.
        if not result.face_blendshapes:
            return FaceScores(found=False)

        # Build a name to score lookup for the first face.
        scores = {b.category_name: b.score for b in result.face_blendshapes[0]}

        # Pull the values needed, averaging paired shapes where it helps.
        jaw_open = scores.get("jawOpen", 0.0)
        smile = (scores.get("mouthSmileLeft", 0.0)
                 + scores.get("mouthSmileRight", 0.0)) / 2.0
        roll_lips = (scores.get("mouthRollUpper", 0.0)
                     + scores.get("mouthRollLower", 0.0)) / 2.0
        brow_raise = (scores.get("browOuterUpLeft", 0.0)
                      + scores.get("browOuterUpRight", 0.0)
                      + scores.get("browInnerUp", 0.0)) / 3.0
        eye_blink = (scores.get("eyeBlinkLeft", 0.0)
                     + scores.get("eyeBlinkRight", 0.0)) / 2.0

        # Return the tidy record.
        return FaceScores(
            found=True,
            jaw_open=jaw_open,
            smile=smile,
            roll_lips=roll_lips,
            brow_raise=brow_raise,
            eye_blink=eye_blink,
        )


# ============================================================================
#  SIMPLE PER-FRAME FACE DETECTORS
# ============================================================================
#  Each takes a FaceScores record and returns True when that action is
#  happening on this frame.
# ============================================================================

def face_open_mouth(scores: FaceScores) -> bool:
    """Mouth open."""

    # Need a face and a jaw-open score over the threshold.
    return scores.found and scores.jaw_open > OPEN_MOUTH_THRESH


def face_smile(scores: FaceScores) -> bool:
    """Smiling."""

    # Need a face and a smile score over the threshold.
    return scores.found and scores.smile > SMILE_THRESH


def face_roll_lips(scores: FaceScores) -> bool:
    """Lips rolled inward."""

    # Need a face and a lip-roll score over the threshold.
    return scores.found and scores.roll_lips > ROLL_LIPS_THRESH


def face_raise_brows(scores: FaceScores) -> bool:
    """Eyebrows raised."""

    # Need a face and a brow-raise score over the threshold.
    return scores.found and scores.brow_raise > BROW_THRESH


# ============================================================================
#  BLINK COUNTER (stateful)
# ============================================================================
#  Blinking once or twice cannot be judged from a single frame; it needs the
#  eyes to close and open again, counted over time. This counter watches the
#  blink score cross the closed threshold and counts each full close-open as
#  one blink.
# ============================================================================

class BlinkCounter:
    """Counts blinks by watching the eyes close and open again."""

    def __init__(self) -> None:
        """Start with no blinks and eyes open."""

        # How many full blinks have been seen.
        self.count = 0

        # Whether the eyes are currently closed, to catch each close once.
        self._eyes_closed = False

    def reset(self) -> None:
        """Clear the count and state for a fresh recording."""

        # Back to zero blinks, eyes open.
        self.count = 0
        self._eyes_closed = False

    def update(self, scores: FaceScores) -> None:
        """Update the blink count from this frame's scores."""

        # No face: treat as eyes open, nothing to count.
        if not scores.found:
            self._eyes_closed = False
            return

        # Eyes are closed when the blink score is high.
        closed_now = scores.eye_blink > BLINK_CLOSED_THRESH

        # Count one blink on the change from open to closed.
        if closed_now and not self._eyes_closed:
            self.count += 1

        # Remember the state for the next frame.
        self._eyes_closed = closed_now