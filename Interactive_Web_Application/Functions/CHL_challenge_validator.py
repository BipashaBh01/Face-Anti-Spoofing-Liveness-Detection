# ============================================================================
#  CHL_challenge_validator.py — validate a recorded challenge sequence
# ============================================================================
#  Takes a challenge recording and the three expected primitives, runs the
#  detectors across every frame, and decides whether the three actions were
#  performed in the correct order and each done correctly.
#
#  Design notes:
#    - Each action must be sustained for a few frames to count, so a one-frame
#      flicker is ignored.
#    - Order is judged by the FIRST sustained occurrence of each action: the
#      first must start before the second, before the third.
#    - Blinks are counted over time, not judged per frame, so a blink action
#      "occurs" at the frame where its required count is reached.
#    - A hand action needs enough valid palm-facing frames; if the hand was
#      never shown properly, the result is UNCLEAR (leads to a re-record with
#      a fresh challenge, so nothing is leaked about which action failed).
# ============================================================================

from __future__ import annotations

# Path handles the recording folder.
from pathlib import Path

# OpenCV reads the saved frames.
import cv2

# The head angle reader (position checker).
from Functions.CHL_position_checker import PositionChecker, PositionConfig

# The face blendshape reader, its detectors and the blink counter.
from Functions.CHL_face_blendshapes import (
    FaceReader,
    face_open_mouth,
    face_smile,
    face_roll_lips,
    face_raise_brows,
    BlinkCounter,
)

# The hand landmarker.
from Functions.CHL_hand_landmarker import HandLandmarker

# The per-frame head and hand detectors.
from Functions.CHL_primitives import HEAD_DETECTORS, HAND_DETECTORS

# The model paths.
from Functions.CHL_config import LANDMARKER_MODEL, HAND_LANDMARKER_MODEL


# ============================================================================
#  SETTINGS
# ============================================================================

# How many consecutive frames an action must hold to count as real.
MIN_SUSTAIN_FRAMES = 3

# Blink score above which the eyes count as closed.
BLINK_CLOSED = 0.50

# Fewest valid palm-facing frames a hand action needs before the recording is
# judged usable for that action. Below this, the attempt is UNCLEAR.
MIN_HAND_VALID_FRAMES = 3


# ============================================================================
#  RESULT
# ============================================================================

# The three possible outcomes of validating a challenge recording.
PASS = "PASS"        # all three actions, correct and in order
FAIL = "FAIL"        # wrong order, missing, or wrong actions
UNCLEAR = "UNCLEAR"  # a hand action could not be read (re-record)


# ============================================================================
#  THE VALIDATOR
# ============================================================================

class ChallengeValidator:
    """Runs the detectors over a recording and checks the action sequence."""

    def __init__(self) -> None:
        """Build the three readers once, so they are reused per recording."""

        # Head angles.
        self._checker = PositionChecker(LANDMARKER_MODEL, PositionConfig())

        # Face blendshapes.
        self._face = FaceReader(LANDMARKER_MODEL)

        # Hand landmarks.
        self._hand = HandLandmarker(HAND_LANDMARKER_MODEL)

    # ------------------------------------------------------------------
    #  Reading one recording into per-frame signals
    # ------------------------------------------------------------------

    def _read_frames(self, frames_dir: Path):
        """Read every saved frame and return the per-frame signals.

        Returns a list, one entry per frame, each a dict with the head
        angles, the face scores, and the hand result.
        """

        # Collect the frame image files in order.
        files = sorted(frames_dir.glob("frame_*.jpg"))

        # Build a per-frame signal record.
        signals = []
        for path in files:

            # Read the image.
            frame = cv2.imread(str(path))
            if frame is None:
                continue

            # Head angles from the position checker.
            pos = self._checker.check(frame)

            # Face blendshape scores.
            face = self._face.read(frame)

            # Hand landmarks and finger states.
            hand = self._hand.check(frame)

            # Whether the thumb points up, for thumbs up.
            thumb_up = (self._hand.thumb_pointing_up(hand.landmarks)
                        if hand.found and hand.landmarks else False)

            # Store the signals for this frame.
            signals.append({
                "yaw": pos.yaw,
                "pitch": pos.pitch,
                "roll": pos.roll,
                "face": face,
                "hand": hand,
                "thumb_up": thumb_up,
            })

        # Hand back all frames' signals.
        return signals

    # ------------------------------------------------------------------
    #  Per-frame active flags for one expected primitive
    # ------------------------------------------------------------------

    def _active_flags(self, signals, key):
        """Return a list of booleans: is this primitive active on each frame.

        Blink primitives are not judged here (they are counted separately),
        so this returns all False for them and the caller handles blinks.
        """

        # Head primitive.
        if key in HEAD_DETECTORS:
            fn = HEAD_DETECTORS[key]
            return [fn(s["yaw"], s["pitch"], s["roll"]) for s in signals]

        # Hand primitive.
        if key in HAND_DETECTORS:
            fn = HAND_DETECTORS[key]
            flags = []
            for s in signals:
                h = s["hand"]
                flags.append(
                    h.found and fn(h.fingers, s["thumb_up"], h.palm_faces_camera)
                )
            return flags

        # Simple face primitives.
        if key == "face_open_mouth":
            return [face_open_mouth(s["face"]) for s in signals]
        if key == "face_smile":
            return [face_smile(s["face"]) for s in signals]
        if key == "face_roll_lips":
            return [face_roll_lips(s["face"]) for s in signals]
        if key == "face_raise_brows":
            return [face_raise_brows(s["face"]) for s in signals]

        # Blink primitives are handled by count, not flags.
        return [False] * len(signals)

    # ------------------------------------------------------------------
    #  Sustained windows from per-frame flags
    # ------------------------------------------------------------------

    def _first_sustained_start(self, flags, after_frame=-1):
        """Return the start frame of the first sustained run of True that
        begins after `after_frame`, or None when there is none.
        """

        # Walk the flags, tracking runs of True.
        run_start = None
        for i, a in enumerate(flags):

            # A run begins.
            if a and run_start is None:
                run_start = i

            # A run ends: check its length and position.
            elif not a and run_start is not None:
                if (i - run_start) >= MIN_SUSTAIN_FRAMES and run_start > after_frame:
                    return run_start
                run_start = None

        # A run reaching the end.
        if run_start is not None:
            if (len(flags) - run_start) >= MIN_SUSTAIN_FRAMES and run_start > after_frame:
                return run_start

        # No qualifying run.
        return None

    # ------------------------------------------------------------------
    #  Blink occurrence frame
    # ------------------------------------------------------------------

    def _blink_occurrence(self, signals, required, after_frame=-1):
        """Return the frame where the blink count reaches `required`,
        counting only from after `after_frame`, or None.
        """

        # Count blinks from just after the previous action.
        counter = BlinkCounter()
        start = after_frame + 1
        for i in range(start, len(signals)):
            counter.update(signals[i]["face"])
            if counter.count >= required:
                return i

        # Never reached the required number.
        return None

    # ------------------------------------------------------------------
    #  Hand validity check
    # ------------------------------------------------------------------

    def _hand_was_readable(self, signals):
        """Return True when the hand was validly shown (palm to camera) on
        enough frames to trust a hand action reading.
        """

        # Count frames with a hand whose palm faced the camera.
        valid = sum(1 for s in signals
                    if s["hand"].found and s["hand"].palm_faces_camera)

        # Enough valid frames means the hand was shown properly.
        return valid >= MIN_HAND_VALID_FRAMES
    

    # ------------------------------------------------------------------
    #  Detection detail: the value and sustained frames at an occurrence
    # ------------------------------------------------------------------

    def _detection_detail(self, signals, key, occ):
        """Return a short string describing the value and sustained frames
        for a detected action, for the debug log.
        """

        # Count how many consecutive frames the action stayed active from
        # its start frame (the sustained run length).
        if key in ("face_blink_once", "face_blink_twice"):
            # Blink is count-based; show the blink score at the frame.
            score = signals[occ]["face"].eye_blink
            return f"blink score {score:.2f}"

        # Build the active flags to measure the sustained run.
        flags = self._active_flags(signals, key)
        held = 0
        for i in range(occ, len(flags)):
            if flags[i]:
                held += 1
            else:
                break

        # Head actions: show the dominant angle at detection.
        if key in HEAD_DETECTORS:
            s = signals[occ]
            if "yaw" in key or "turn" in key:
                val = s["yaw"]
            elif "tilt" in key:
                val = s["roll"]
            else:  # look up/down
                val = s["pitch"]
            return f"held {held} frames, angle {val:+.1f} deg"

        # Face actions: show the blendshape score (a real 0-1 confidence).
        face = signals[occ]["face"]
        face_score = {
            "face_open_mouth":  face.jaw_open,
            "face_smile":       face.smile,
            "face_roll_lips":   face.roll_lips,
            "face_raise_brows": face.brow_raise,
        }.get(key)
        if face_score is not None:
            return f"held {held} frames, score {face_score:.2f}"

        # Hand actions: the gesture is a pattern match, so show frames held
        # and the palm-facing state (the trust signal for hands).
        if key in HAND_DETECTORS:
            palm = signals[occ]["hand"].palm_faces_camera
            return f"held {held} frames, palm-facing {palm}"

        # Fallback.
        return f"held {held} frames"


    # ------------------------------------------------------------------
    #  Main entry: validate a recording against the expected primitives
    # ------------------------------------------------------------------

    def validate(self, frames_dir, expected_keys) -> str:
        """Validate one challenge recording.

        Parameters
        ----------
        frames_dir : path to the folder holding the saved frames.
        expected_keys : list of three primitive keys, in the required order.

        Returns
        -------
        str : PASS, FAIL or UNCLEAR.
        """

        # Read all frames into per-frame signals.
        signals = self._read_frames(Path(frames_dir))

        # No frames at all: cannot judge, treat as unclear to re-record.
        if not signals:
            print("Validator: no frames read.")
            detail = {"actions": expected_keys, "outcomes": [], "details": []}
            return UNCLEAR, detail

        # Track the frame the previous action occurred at, so each action is
        # found after the one before it.
        prev_frame = -1

        # Per-action records for the CSV: the outcome (detected/not_found) and
        # the detail string (score/angle + frames) for each expected action.
        action_outcomes = []
        action_details = []

        # Header for the per-action detection log.
        print("-" * 60)
        print("CHALLENGE VALIDATION")
        print(f"  Expected sequence : {expected_keys}")
        print(f"  Frames read       : {len(signals)}")

        # Work through the three expected actions in order.
        for position, key in enumerate(expected_keys):

            # ---- blink actions: count-based ----
            if key == "face_blink_once":
                occ = self._blink_occurrence(signals, 1, prev_frame)
            elif key == "face_blink_twice":
                occ = self._blink_occurrence(signals, 2, prev_frame)

            # ---- hand actions: need a readable hand first ----
            elif key in HAND_DETECTORS:

                # If the hand was never shown properly, the attempt is
                # unclear, not a fail, so a fresh challenge is given.
                if not self._hand_was_readable(signals):
                    print(f"  [{position}] {key:20} -> HAND NOT READABLE (unclear)")
                    print("  RESULT: UNCLEAR (re-record)")
                    print("-" * 60)
                    # Record what was known before returning.
                    action_outcomes.append("unclear")
                    action_details.append("hand not readable")
                    detail = {
                        "actions": expected_keys,
                        "outcomes": action_outcomes,
                        "details": action_details,
                    }
                    return UNCLEAR, detail

                # Otherwise find its first sustained occurrence in order.
                flags = self._active_flags(signals, key)
                occ = self._first_sustained_start(flags, prev_frame)

            # ---- head and simple face actions: window-based ----
            else:
                flags = self._active_flags(signals, key)
                occ = self._first_sustained_start(flags, prev_frame)

            # Log this action's outcome.
            if occ is None:
                print(f"  [{position}] {key:20} -> NOT FOUND after frame {prev_frame}")
                print("  RESULT: FAIL")
                print("-" * 60)
                # This action was not found; record it and stop.
                action_outcomes.append("not_found")
                action_details.append("—")
                detail = {
                    "actions": expected_keys,
                    "outcomes": action_outcomes,
                    "details": action_details,
                }
                return FAIL, detail

            # Found in order: log the frame plus the value and frames held.
            detail_str = self._detection_detail(signals, key, occ)
            print(f"  [{position}] {key:20} -> detected at frame {occ}  ({detail_str})")

            # Record this action as detected, with its detail string.
            action_outcomes.append("detected")
            action_details.append(detail_str)

            # Advance the marker so the next action must come after this one.
            prev_frame = occ

        # All three actions were found, correct and in order.
        print("  RESULT: PASS (all three correct and in order)")
        print("-" * 60)
        detail = {
            "actions": expected_keys,
            "outcomes": action_outcomes,
            "details": action_details,
        }
        return PASS, detail