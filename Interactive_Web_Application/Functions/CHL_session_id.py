# ============================================================================
#  CHL_session_id.py — builds the id used to name each recording folder
# ============================================================================
#  The id has the form:  YYYYMMDD_HHMMSS_N
#
#    YYYYMMDD   the date, for example 20260128
#    HHMMSS     the time, for example 143015
#    N          a counter that starts at 1 and grows with each recording,
#               resetting to 1 at the start of every new hour
#
#  Example ids:
#    20260128_143015_1   first recording in the 2 PM hour
#    20260128_143847_2   second recording in the same hour
#    20260128_150210_1   new hour, so the counter resets to 1
# ============================================================================

from __future__ import annotations

# datetime gives the current date and time.
from datetime import datetime


# ============================================================================
#  COUNTER
# ============================================================================

class SessionIdMaker:
    """Creates recording ids and resets the counter each new hour."""

    def __init__(self):
        """Start with no counter and no recorded hour yet."""

        # The hour that the counter currently belongs to, as text.
        # None means no recording has been made yet.
        self._current_hour: str | None = None

        # The last number given out within the current hour.
        self._counter = 0

    # ------------------------------------------------------------------------

    def new_id(self, now: datetime | None = None) -> str:
        """Build and return the next recording id."""

        # Use the real current time unless one is passed in for testing.
        now = datetime.now() if now is None else now

        # Text for the current hour, for example "20260128_14".
        # Two ids share an hour only when this text matches.
        hour_key = now.strftime("%Y%m%d_%H")

        # When the hour has changed, reset the counter to zero.
        if hour_key != self._current_hour:

            # Remember the new hour.
            self._current_hour = hour_key

            # Reset the counter for the new hour.
            self._counter = 0

        # Move to the next number within this hour.
        self._counter += 1

        # Date and time parts, for example "20260128_143015".
        date_time = now.strftime("%Y%m%d_%H%M%S")

        # Join the date, time and counter into the final id.
        return f"{date_time}_{self._counter}"