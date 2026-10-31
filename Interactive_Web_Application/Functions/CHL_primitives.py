# ============================================================================
#  CHL_primitives.py — per-frame detectors for the challenge primitives
# ============================================================================
#  Each detector answers a simple question about ONE frame: is this action
#  happening right now? A later stage runs these across the whole challenge
#  recording to work out the order the actions occurred in.
#
#  This file is built group by group:
#    Group 1  head  (this step)
#    Group 2  face  (next)
#    Group 3  hand  (after)
# ============================================================================

from __future__ import annotations

# math is used for angle checks and NaN handling.
import math


# ============================================================================
#  THRESHOLDS
# ============================================================================
#  Head actions fire when an angle passes a clear threshold. These are set a
#  little beyond normal small movement so a deliberate action is needed, and
#  can be tuned after real testing.
# ============================================================================

# How many degrees of yaw counts as a deliberate left or right turn.
HEAD_TURN_DEGREES = 20.0

# How many degrees of roll counts as a deliberate sideways tilt.
HEAD_TILT_DEGREES = 18.0

# How many degrees of pitch counts as a deliberate look up or down.
HEAD_PITCH_DEGREES = 15.0


# ============================================================================
#  ANGLE SIGN CONVENTIONS
# ============================================================================
#  These follow the same matrix-to-angle output used elsewhere in the app:
#    yaw   : positive when the face turns one way, negative the other
#    pitch : positive up, negative down (checked against real data below)
#    roll  : positive one tilt, negative the other
#
#  The exact sign for each direction is confirmed against real readings when
#  testing, and flipped here if needed. The mirror is handled at display time,
#  not here; these work on the real, unmirrored angles.
# ============================================================================


# ============================================================================
#  HEAD DETECTORS (Group 1)
# ============================================================================
#  Each takes the frame's yaw, pitch and roll in degrees and returns True
#  when that head action is happening on this frame. Angles may be NaN when
#  no face or pose was found, which always returns False.
# ============================================================================

def _valid(*angles) -> bool:
    """Return True only when every angle is a real number, not NaN."""

    # Any NaN means the pose was not measured, so the action cannot fire.
    return all(not math.isnan(a) for a in angles)


def head_turn_left(yaw, pitch, roll) -> bool:
    """Head turned to the user's left."""

    # Need a valid yaw.
    if not _valid(yaw):
        return False

    # A left turn shows as a positive yaw in this setup, confirmed by real
    # testing. It must also be the dominant movement, so a tilt does not
    # trigger it.
    return yaw > HEAD_TURN_DEGREES and abs(yaw) > abs(roll)


def head_turn_right(yaw, pitch, roll) -> bool:
    """Head turned to the user's right."""

    # Need a valid yaw.
    if not _valid(yaw):
        return False

    # A right turn shows as a negative yaw here. It must also be the dominant
    # movement, so a tilt does not trigger it.
    return yaw < -HEAD_TURN_DEGREES and abs(yaw) > abs(roll)


def head_tilt_left(yaw, pitch, roll) -> bool:
    """Head tilted sideways to the user's left."""

    # Need valid angles.
    if not _valid(yaw, pitch, roll):
        return False

    # A left tilt is a negative roll past the tilt threshold, and roll must
    # be the dominant movement so a turn or look is not misread as a tilt.
    return (roll < -HEAD_TILT_DEGREES
            and abs(roll) > abs(yaw)
            and abs(roll) > abs(pitch))


def head_tilt_right(yaw, pitch, roll) -> bool:
    """Head tilted sideways to the user's right."""

    # Need valid angles.
    if not _valid(yaw, pitch, roll):
        return False

    # A right tilt is a positive roll past the tilt threshold, and roll must
    # be the dominant movement so a turn or look is not misread as a tilt.
    return (roll > HEAD_TILT_DEGREES
            and abs(roll) > abs(yaw)
            and abs(roll) > abs(pitch))


def head_look_down(yaw, pitch, roll) -> bool:
    """Head tipped down / looking down."""

    # Need a valid pitch.
    if not _valid(pitch):
        return False

    # Looking down shows as a positive pitch here, confirmed by testing. It
    # must also be the dominant movement, so a tilt does not trigger it.
    return pitch > HEAD_PITCH_DEGREES and abs(pitch) > abs(roll)


def head_look_up(yaw, pitch, roll) -> bool:
    """Head tipped up / looking up."""

    # Need a valid pitch.
    if not _valid(pitch):
        return False

    # Looking up shows as a negative pitch here. It must also be the dominant
    # movement, so a tilt does not trigger it.
    return pitch < -HEAD_PITCH_DEGREES and abs(pitch) > abs(roll)


# ============================================================================
#  REGISTRY (head only for now)
# ============================================================================
#  Maps each primitive key to the function that detects it on one frame. Face
#  and hand entries are added in the next groups.
# ============================================================================

HEAD_DETECTORS = {
    "head_turn_left":  head_turn_left,
    "head_turn_right": head_turn_right,
    "head_tilt_left":  head_tilt_left,
    "head_tilt_right": head_tilt_right,
    "head_look_down":  head_look_down,
    "head_look_up":    head_look_up,
}


# ============================================================================
#  HAND DETECTORS (Group 3)
# ============================================================================
#  Each takes the finger-state list, the thumb-up flag, and whether the palm
#  faces the camera. Most gestures only trust the finger states when the palm
#  faces the camera flat, so a turned or edge-on hand returns False (blank).
#  Thumbs up is the exception: it is naturally shown with the palm sideways,
#  so it does not require a camera-facing palm.
# ============================================================================

def hand_one_finger(fingers, thumb_up, palm_faces) -> bool:
    """Only the index finger is extended, palm facing the camera."""

    # Need a camera-facing palm for a trustworthy finger count.
    if not palm_faces:
        return False

    # Index up, everything else down.
    return fingers == [False, True, False, False, False]


def hand_peace(fingers, thumb_up, palm_faces) -> bool:
    """Index and middle extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers == [False, True, True, False, False]


def hand_three(fingers, thumb_up, palm_faces) -> bool:
    """Three fingers extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers in (
        [False, True, True, True, False],   # index, middle, ring
        [True, True, True, False, False],   # thumb, index, middle
    )


def hand_four(fingers, thumb_up, palm_faces) -> bool:
    """Four fingers extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers == [False, True, True, True, True]


def hand_five(fingers, thumb_up, palm_faces) -> bool:
    """All five fingers extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers == [True, True, True, True, True]


def hand_fist(fingers, thumb_up, palm_faces) -> bool:
    """No fingers extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers == [False, False, False, False, False]


def hand_thumbs_up(fingers, thumb_up, palm_faces) -> bool:
    """Only the thumb extended and pointing up.

    Thumbs up is shown with the palm sideways, so it does not require a
    camera-facing palm; the thumb-up direction and folded fingers are enough.
    """

    return (fingers[0]
            and not any(fingers[1:])
            and thumb_up)


def hand_rock(fingers, thumb_up, palm_faces) -> bool:
    """Index and pinky extended, palm facing the camera."""

    if not palm_faces:
        return False
    return fingers == [False, True, False, False, True]


# ============================================================================
#  HAND REGISTRY
# ============================================================================

HAND_DETECTORS = {
    "hand_one_finger": hand_one_finger,
    "hand_peace":      hand_peace,
    "hand_three":      hand_three,
    "hand_four":       hand_four,
    "hand_five":       hand_five,
    "hand_fist":       hand_fist,
    "hand_thumbs_up":  hand_thumbs_up,
    "hand_rock":       hand_rock,
}