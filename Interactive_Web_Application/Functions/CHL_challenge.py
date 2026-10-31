# ============================================================================
#  CHL_challenge.py — the challenge primitives and a random picker
# ============================================================================
#  Holds the full list of challenge primitives and picks three at random for
#  each attempt. Detection of whether they were performed is built later; this
#  file only names the primitives and chooses a sequence.
# ============================================================================

from __future__ import annotations

# random chooses the three primitives for each challenge.
import random


# ============================================================================
#  THE 20 PRIMITIVES
# ============================================================================
#  Each primitive has:
#    key      : a short unique id used in code and messages
#    text     : the instruction shown to the user
#    group    : head, face or hand (hand needs the hand landmarker)
# ============================================================================

PRIMITIVES = [

    # ---- head (6) ----
    {"key": "head_turn_left",  "text": "Turn your head left",  "group": "head"},
    {"key": "head_turn_right", "text": "Turn your head right", "group": "head"},
    {"key": "head_tilt_left",  "text": "Tilt your head left",  "group": "head"},
    {"key": "head_tilt_right", "text": "Tilt your head right", "group": "head"},
    {"key": "head_look_down",  "text": "Look down",            "group": "head"},
    {"key": "head_look_up",    "text": "Look up",              "group": "head"},

    # ---- face (6) ----
    {"key": "face_blink_once",   "text": "Blink once",         "group": "face"},
    {"key": "face_blink_twice",  "text": "Blink twice",        "group": "face"},
    {"key": "face_roll_lips",    "text": "Roll your lips in",  "group": "face"},
    {"key": "face_open_mouth",   "text": "Open your mouth",    "group": "face"},
    {"key": "face_smile",        "text": "Smile",              "group": "face"},
    {"key": "face_raise_brows",  "text": "Raise your eyebrows","group": "face"},

    # ---- hand (8) — need the hand landmarker ----
    {"key": "hand_one_finger",  "text": "Show one finger",     "group": "hand"},
    {"key": "hand_peace",       "text": "Show a peace sign",   "group": "hand"},
    {"key": "hand_three",       "text": "Show three fingers",  "group": "hand"},
    {"key": "hand_four",        "text": "Show four fingers",   "group": "hand"},
    {"key": "hand_five",        "text": "Show five fingers",   "group": "hand"},
    {"key": "hand_fist",        "text": "Make a fist",         "group": "hand"},
    {"key": "hand_thumbs_up",   "text": "Thumbs up",           "group": "hand"},
    {"key": "hand_rock",        "text": "Show a rock sign",    "group": "hand"},
]


# A quick lookup from key to the full primitive record.
PRIMITIVE_BY_KEY = {item["key"]: item for item in PRIMITIVES}


# ============================================================================
#  PICK A CHALLENGE
# ============================================================================

def pick_challenge(count: int = 3) -> list[dict]:
    """Pick a random ordered challenge of distinct primitives.

    Parameters
    ----------
    count : int
        How many primitives to pick. Three by design.

    Returns
    -------
    list of dict
        The chosen primitives in order, each the full record.
    """

    # Choose distinct primitives, keeping the order they are drawn in.
    return random.sample(PRIMITIVES, count)


def challenge_needs_hand(challenge: list[dict]) -> bool:
    """Return True when any primitive in the challenge is a hand gesture."""

    # A hand gesture means the hand landmarker must run on the recording.
    return any(item["group"] == "hand" for item in challenge)