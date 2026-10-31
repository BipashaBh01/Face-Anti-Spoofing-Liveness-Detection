# ============================================================================
#  CHL_position_checker.py — checks the face position against the camera
# ============================================================================
#  Runs MediaPipe on one camera frame, measures where the face sits, and
#  returns a single short instruction such as "Move a little right" when
#  the position needs correcting, or a pass when it is correct.
#
#  Only measurement and judgement live here. Sending frames and showing
#  messages happen elsewhere, so thresholds can change without touching
#  the server or the browser.
# ============================================================================

from __future__ import annotations

# Standard maths module, used for angle conversion and NaN checks.
import math

# dataclass builds small container classes without boilerplate.
from dataclasses import dataclass

# Path handles the model file location.
from pathlib import Path

# OpenCV converts colour order and is already used across the project.
import cv2

# NumPy holds the frame as an array.
import numpy as np

# MediaPipe provides the face landmark model.
import mediapipe as mp

# The Tasks API pieces needed to build the landmarker.
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================================
#  CONFIGURATION
# ============================================================================

@dataclass
class PositionConfig:
    """Threshold values that decide whether a face is well positioned."""

    # Largest allowed left/right head turn, in degrees.
    max_yaw_degrees: float = 18.0

    # Largest allowed up/down head tilt, in degrees.
    max_pitch_degrees: float = 20.0

    # Largest allowed sideways head tilt, in degrees.
    max_roll_degrees: float = 20.0

    # Smallest allowed face height as a fraction of the frame height.
    # Below this the person is too far away. Tightened so the face must
    # fill the on-screen oval before recording starts.
    min_face_height_frac: float = 0.42

    # Largest allowed face height as a fraction of the frame height.
    # Above this the person is too close. Tightened to match the oval.
    max_face_height_frac: float = 0.62

    # Largest allowed sideways shift of the face from the centre,
    # as a fraction of the frame width. Tightened so the face must be
    # centred within the oval.
    max_offset_x_frac: float = 0.10

    # Largest allowed up/down shift of the face from the centre,
    # as a fraction of the frame height. Tightened to match the oval.
    max_offset_y_frac: float = 0.10

    # Confidence floor passed to the detector.
    min_detection_confidence: float = 0.4

    # Smallest allowed average brightness of the face (0..255 scale).
    min_face_brightness: float = 45.0

    # Largest allowed average brightness of the face (0..255 scale).
    max_face_brightness: float = 225.0

    # Largest allowed brightness difference between the left and right
    # halves of the face.
    max_lighting_imbalance: float = 55.0


# ============================================================================
#  RESULT
# ============================================================================

@dataclass
class PositionResult:
    """Outcome of checking one frame."""
 
    # True when every check passed.
    ok: bool
 
    # Short instruction shown to the person.
    message: str
 
    # Number of faces found in the frame.
    face_count: int = 0
 
    # Face centre in the 0..1 range. 0.5, 0.5 is the middle.
    centre_x: float = 0.5
    centre_y: float = 0.5
 
    # Head angles in degrees. NaN when the orientation is unknown.
    yaw: float = float("nan")
    pitch: float = float("nan")
    roll: float = float("nan")

    # Gap between the face box and each frame edge, as a fraction. Large
    # means the face sits well inside on that side; small means it is near
    # that edge and starting to leave. Default 1.0 means safely inside.
    gap_left: float = 1.0
    gap_right: float = 1.0
    gap_top: float = 1.0
    gap_bottom: float = 1.0
    


# ============================================================================
#  CHECKER
# ============================================================================

class PositionChecker:
    """Detects the face and judges its position against the thresholds."""

    def __init__(self, model_path: Path, config: PositionConfig | None = None):
        """Build the MediaPipe landmarker once, ready to reuse per frame."""

        # Keep the thresholds, using defaults when none are supplied.
        self.config = config or PositionConfig()

        # Stop early with a clear message when the model file is missing.
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}"
            )

        # Point MediaPipe at the downloaded model file.
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))

        # Configure the landmarker: one image at a time, one face, with the
        # transformation matrix that gives head orientation.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_detection_confidence,
            output_facial_transformation_matrixes=True,
        )

        # Create the landmarker instance from those options.
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------------

    def check(self, frame_bgr: np.ndarray) -> PositionResult:
        """Check one frame and return the instruction for the person."""

        # Read the frame height and width in pixels.
        frame_height, frame_width = frame_bgr.shape[:2]

        # MediaPipe expects RGB, so convert from OpenCV's BGR order.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wrap the array in the MediaPipe image type.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection on the frame.
        detection = self.landmarker.detect(mp_image)

        # Count how many faces were detected.
        face_count = len(detection.face_landmarks)

        # When no face is found, ask the person to come into view.
        if face_count == 0:
            return PositionResult(ok=False, message="Please move in front of the camera")

        # When more than one face is found, ask for a single person.
        if face_count > 1:
            return PositionResult(ok=False, message="Only one person should be visible", face_count=face_count)

        # Take the landmarks of the single detected face.
        landmarks = detection.face_landmarks[0]

        # Collect the x and y of every landmark as arrays in 0..1 range.
        xs = np.array([point.x for point in landmarks], dtype=np.float32)
        ys = np.array([point.y for point in landmarks], dtype=np.float32)

        # Gap between the face box and each frame edge, as a fraction.
        # A small gap means the face is near that edge and part of it is
        # leaving the frame on that side. Each side is kept separately so
        # the direction of the movement can be named.
        gap_left = float(xs.min())          # space to the left edge (0)
        gap_right = float(1.0 - xs.max())   # space to the right edge (1)
        gap_top = float(ys.min())           # space to the top edge (0)
        gap_bottom = float(1.0 - ys.max())  # space to the bottom edge (1)

        # Face box edges in the 0..1 range.
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        # Face height as a fraction of the frame height.
        height_fraction = y_max - y_min

        # Face centre position in the 0..1 range.
        centre_x = (x_min + x_max) / 2.0
        centre_y = (y_min + y_max) / 2.0


        # Shift of the face centre from the frame centre (0.5, 0.5).
        offset_x = centre_x - 0.5
        offset_y = centre_y - 0.5

        # Read head orientation from the transformation matrix when present.
        if detection.facial_transformation_matrixes:
            yaw, pitch, roll = self._matrix_to_angles(
                detection.facial_transformation_matrixes[0]
            )
        else:
            # Without a matrix the orientation is unknown; mark as NaN.
            yaw = pitch = roll = float("nan")


        # Build a result that carries the measured values, so callers like
        # the recording monitor receive the current face data every time.
        def make(ok: bool, message: str) -> PositionResult:
            return PositionResult(
                ok=ok,
                message=message,
                face_count=face_count,
                centre_x=centre_x,
                centre_y=centre_y,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                gap_left=gap_left,
                gap_right=gap_right,
                gap_top=gap_top,
                gap_bottom=gap_bottom,
            )
        
        # ---- run the checks in priority order, one message at a time ----

        # Distance first: too far away.
        if height_fraction < self.config.min_face_height_frac:
            return make(ok=False, message="Please move closer to the camera")

        # Distance: too close.
        if height_fraction > self.config.max_face_height_frac:
            return make(ok=False, message="Please move back a little")

        # Sideways position: face sits left of centre.
        if offset_x < -self.config.max_offset_x_frac:
            return make(ok=False, message="Please move a little to the left")

        # Sideways position: face sits right of centre.
        if offset_x > self.config.max_offset_x_frac:
            return make(ok=False, message="Please move a little to the right")

        # Vertical position: face sits above centre.
        if offset_y < -self.config.max_offset_y_frac:
            return make(ok=False, message="Please move down slightly")

        # Vertical position: face sits below centre.
        if offset_y > self.config.max_offset_y_frac:
            return make(ok=False, message="Please move up slightly")

        # Orientation: only checked when the angles are known.
        if not math.isnan(yaw):

            # Turned too far left or right.
            if abs(yaw) > self.config.max_yaw_degrees:
                return make(ok=False, message="Please look straight at the camera")

            # Tilted too far up or down.
            if abs(pitch) > self.config.max_pitch_degrees:
                return make(ok=False, message="Please keep your head level")

            # Tilted sideways too far.
            if abs(roll) > self.config.max_roll_degrees:
                return make(ok=False, message="Please keep your head upright")
        
        # ---- lighting checks ----

        # Convert the frame to the LAB colour space; the L channel is
        # perceived brightness, not skewed by colour.
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        # Face box edges in pixels, from the 0..1 landmark range.
        px_x_min = int(x_min * frame_width)
        px_x_max = int(x_max * frame_width)
        px_y_min = int(y_min * frame_height)
        px_y_max = int(y_max * frame_height)

        # Cut out the face region from the L (brightness) channel.
        face_l = lab[px_y_min:px_y_max, px_x_min:px_x_max, 0]

        # Only run lighting checks when the region is non-empty.
        if face_l.size > 0:

            # Average brightness across the whole face.
            brightness = float(np.mean(face_l))

            # Too dark overall.
            if brightness < self.config.min_face_brightness:
                return make(ok=False, message="Please move to brighter light")

            # Too bright overall.
            if brightness > self.config.max_face_brightness:
                return make(ok=False, message="Please reduce direct light")

            # Split the face into left and right halves.
            mid = face_l.shape[1] // 2
            left_half = face_l[:, :mid]
            right_half = face_l[:, mid:]

            # Only compare when both halves have pixels.
            if left_half.size > 0 and right_half.size > 0:

                # Brightness difference between the two sides.
                imbalance = abs(float(np.mean(left_half)) - float(np.mean(right_half)))

                # One side notably darker than the other.
                if imbalance > self.config.max_lighting_imbalance:
                    return make(ok=False, message="Please face the light evenly")

        # Every check passed: the position is good.
        return make(ok=True, message="Please hold till a 20s video gets captured...")

    # ------------------------------------------------------------------------

    @staticmethod
    def _matrix_to_angles(matrix) -> tuple[float, float, float]:
        """Turn the 4x4 face matrix into yaw, pitch and roll in degrees."""

        # Take the top-left 3x3 rotation part of the matrix.
        rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]

        # Helper length used to detect the unstable straight-up case.
        sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)

        # Normal case: angles are well defined.
        if sy >= 1e-6:
            pitch = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = math.atan2(rotation[1, 0], rotation[0, 0])
        # Rare straight-up case: fall back to a stable estimate.
        else:
            pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
            yaw = math.atan2(-rotation[2, 0], sy)
            roll = 0.0

        # Convert all three from radians to degrees before returning.
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)