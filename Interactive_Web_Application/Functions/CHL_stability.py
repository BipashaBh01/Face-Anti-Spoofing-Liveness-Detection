# ============================================================================
#  CHL_stability.py — tracks a continuous position hold before recording
# ============================================================================
#  The position checker judges one frame at a time. This file tracks the
#  result across many frames, to confirm the person holds a correct
#  position for a set number of seconds before recording starts.
#
#  A single correct frame is not enough: a person could face the camera for
#  an instant and turn away. Requiring a continuous hold prevents that. Any
#  incorrect frame resets the hold, so the full time must pass unbroken.
#
#  The timing is based on the clock, not on a count of frames, so it stays
#  accurate whatever the frame rate happens to be.
# ============================================================================

from __future__ import annotations

# time.monotonic gives a steadily increasing clock, safe for measuring
# durations even if the system clock changes.
import time

# dataclass builds the small result container without boilerplate.
from dataclasses import dataclass


# ============================================================================
#  RESULT
# ============================================================================

@dataclass
class StabilityState:
    """Outcome of one stability update, describing the current phase."""

    # The stage of the hold. One of:
    #   "checking"        position is not yet correct
    #   "holding"         position is correct and the countdown is running
    #   "start_recording" the full hold has completed
    phase: str

    # Whole seconds left in the countdown while holding (3, then 2, then 1).
    # Zero at every other time.
    countdown: int = 0


# ============================================================================
#  TRACKER
# ============================================================================

class StabilityTracker:
    """Measures how long a correct position has been held without breaking."""

    def __init__(self, hold_seconds: float = 3.0):
        """Set up the tracker with the required hold length in seconds."""

        # How long the position must stay correct before recording.
        self.hold_seconds = hold_seconds

        # The clock time when the current unbroken hold began.
        # None means no hold is in progress right now.
        self._hold_started_at: float | None = None

        # True once the full hold has completed, so it is only reported once.
        self._completed = False

    # ------------------------------------------------------------------------

    def update(self, position_ok: bool, now: float | None = None) -> StabilityState:
        """Take one frame's position result and return the current phase."""

        # Use the real clock unless a time is passed in for testing.
        now = time.monotonic() if now is None else now

        # ---- position is not correct ----
        if not position_ok:

            # Cancel any hold in progress.
            self._hold_started_at = None

            # Clear the completed flag so a fresh hold can start later.
            self._completed = False

            # Report that the setup is still being checked.
            return StabilityState(phase="checking", countdown=0)

        # ---- position is correct ----

        # Start the hold timer if this is the first correct frame.
        if self._hold_started_at is None:
            self._hold_started_at = now

        # Work out how long the position has been held so far.
        held = now - self._hold_started_at

        # ---- the full hold has completed ----
        if held >= self.hold_seconds:

            # Mark it completed so the recording trigger is clear.
            self._completed = True

            # Report that recording should start now.
            return StabilityState(phase="start_recording", countdown=0)

        # ---- still holding: work out the countdown number ----

        # Seconds remaining in the hold.
        remaining = self.hold_seconds - held

        # Round up so the legend reads 3, then 2, then 1.
        countdown = int(remaining) + 1

        # Never show more than the starting number.
        countdown = min(countdown, int(self.hold_seconds))

        # Report that the hold is running with this countdown value.
        return StabilityState(phase="holding", countdown=countdown)

    # ------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all progress, used when starting a fresh attempt."""

        # Forget any hold in progress.
        self._hold_started_at = None

        # Clear the completed flag.
        self._completed = False

    # ------------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True once the full hold has been reached."""

        # Simply report the stored flag.
        return self._completed