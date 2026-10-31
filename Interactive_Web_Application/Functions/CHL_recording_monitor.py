# ============================================================================
#  CHL_recording_monitor.py — watches the person while recording is running
# ============================================================================
#  During the 20 second recording, each frame is watched for problems that
#  should pause the recording. Small natural movements are allowed; only a
#  real problem that lasts long enough causes a pause.
#
#  Checks performed here:
#    Job A  position drift   person drifts too far from a good position
#    (1)    no face          the face leaves the frame
#    (2)    multiple faces   a second person appears
#    (6)    sudden jump      the face position jumps far between frames
#
#  Face-swap detection (same person or not) is handled in a separate file,
#  because it needs a face recognition model rather than landmarks.
#
#  A problem must last a short time before pausing, so a brief blink, a
#  small head move, or one noisy frame does not stop the recording.
# ============================================================================

from __future__ import annotations

# time.monotonic measures how long a problem has lasted.
import time

# math is used for angle limits.
import math

# dataclass builds the small config and result containers.
from dataclasses import dataclass


# ============================================================================
#  CONFIGURATION
# ============================================================================

@dataclass
class MonitorConfig:
    """Limits that decide when recording should pause."""

    # ---- Job A: looser position limits than the entry check ----

    # Largest allowed left/right head turn during recording, in degrees.
    max_yaw_degrees: float = 23.0

    # Largest allowed up/down head tilt during recording, in degrees.
    max_pitch_degrees: float = 23.0

    # Largest allowed sideways head tilt during recording, in degrees.
    max_roll_degrees: float = 20.0

    # Largest sideways move allowed from the starting position, as a
    # fraction of frame width. Measured from where the face sat when
    # recording began, so it adapts to each user. A deliberate left or
    # right move beyond this pauses, before face parts leave the frame.
    max_move_horizontal: float = 0.07

    # Largest up or down move allowed from the starting position, as a
    # fraction of frame height. Vertical framing has more room, so this is
    # looser than the horizontal limit.
    max_move_vertical: float = 0.10

    # ---- sudden jump ----

    # Largest allowed movement of the face centre between two frames,
    # as a fraction of the frame size. A larger jump looks like a
    # spliced or swapped video rather than natural movement.
    max_jump_frac: float = 0.25

    # ---- timing ----

    # How long a problem must last before the recording pauses, in seconds.
    # Short problems (a blink, one noisy frame) are ignored.
    pause_after_seconds: float = 1.2

    # Total time the face may spend moved away from its starting position
    # before pausing, in seconds. Adds up across the whole recording, so a
    # slow continuous drift is caught as well as quick moves.
    moved_time_limit_seconds: float = 2.5

    # How many separate movements away from the start are allowed. Exceeding
    # this pauses, so the fifth distinct move stops a flick-pause-flick
    # pattern.
    move_count_limit: int = 4

    # How far the face centre may sit from its start before it counts as
    # moved, as a fraction of the frame. Small natural jitter stays under.
    move_pos_margin: float = 0.04

    # How far a head angle (yaw, pitch or roll) may change from its start
    # before it counts as moved, in degrees. Covers nods and tilts.
    move_angle_margin: float = 8.0

    



# ============================================================================
#  RESULT
# ============================================================================

@dataclass
class MonitorResult:
    """Outcome of watching one frame during recording."""

    # True when the recording should pause now.
    should_pause: bool

    # Short reason for the pause, shown to the person.
    # Example: "You moved right". Empty when not pausing.
    reason: str = ""


# ============================================================================
#  MONITOR
# ============================================================================

class RecordingMonitor:
    """Watches frames during recording and decides when to pause."""

    def __init__(self, config: MonitorConfig | None = None):
        """Set up the monitor with its limits and starting state."""

        # Keep the limits, using defaults when none are supplied.
        self.config = config or MonitorConfig()

        # The face centre seen in the previous frame, to measure a jump.
        # None until the first face is seen.
        self._last_centre: tuple[float, float] | None = None

        # The clock time when the current problem began.
        # None means there is no problem in progress.
        self._problem_started_at: float | None = None

        # The face centre at the moment recording began, used as the
        # baseline that movement is measured against. None until the first
        # good frame sets it.
        self._baseline_centre: tuple[float, float] | None = None

        # The face position and angles captured at the start of recording,
        # used as the baseline that all movement is measured against.
        self._move_base: tuple[float, float, float, float, float] | None = None

        # Total time the face has spent moved away from the baseline.
        self._moved_time: float = 0.0

        # The time of the previous frame, to add up moved-time.
        self._move_prev_t: float | None = None

        # How many separate movements away from the baseline have happened.
        self._move_count: int = 0

        # Whether the face is currently away from the baseline, so each move
        # is counted once, not every frame it stays away.
        self._move_away: bool = False


    # ------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state, used when a fresh recording begins."""

        # Forget the previous face position.
        self._last_centre = None

        # Forget any problem in progress.
        self._problem_started_at = None

        # Forget the starting position, so the next recording sets a fresh
        # baseline from its own first good frame.
        self._baseline_centre = None

        # Forget the movement baseline and all movement tracking.
        self._move_base = None
        self._moved_time = 0.0
        self._move_prev_t = None
        self._move_count = 0
        self._move_away = False


    # ------------------------------------------------------------------------

    def update(
        self,
        face_count: int,
        centre_x: float,
        centre_y: float,
        yaw: float,
        pitch: float,
        roll: float,
        now: float | None = None,
    ) -> MonitorResult:
        """Watch one frame and decide whether to pause.

        face_count : number of faces detected in the frame
        centre_x   : face centre x in the 0..1 range
        centre_y   : face centre y in the 0..1 range
        yaw, pitch, roll : head angles in degrees (NaN when unknown)
        """

        # Use the real clock unless a time is passed in for testing.
        now = time.monotonic() if now is None else now

        # Find out if this frame has a problem, and the reason if so.
        reason = self._find_problem(face_count, centre_x, centre_y, yaw, pitch, roll)

        
        # ---- movement check: any drift from the starting position ----
        # The face position and angles when recording began are the
        # baseline. Any move away from it counts as movement, however slow.
        # Two ways to pause: too much total moved-time (catches slow
        # continuous drift), or too many separate movements (catches
        # repeated flick-pause moves).
        if self._move_base is None:

            # Capture the baseline on the first frame with a face.
            self._move_base = (centre_x, centre_y, yaw, pitch, roll)
            self._move_prev_t = now

        else:

            # Unpack the baseline position and angles.
            base_x, base_y, base_yaw, base_pitch, base_roll = self._move_base

            # Moved if the centre sits beyond the position margin.
            moved = (abs(centre_x - base_x) > self.config.move_pos_margin
                     or abs(centre_y - base_y) > self.config.move_pos_margin)

            # Also moved if any head angle changed beyond the angle margin.
            if not math.isnan(yaw) and not math.isnan(base_yaw):
                if (abs(yaw - base_yaw) > self.config.move_angle_margin
                        or abs(pitch - base_pitch) > self.config.move_angle_margin
                        or abs(roll - base_roll) > self.config.move_angle_margin):
                    moved = True

            # Add the gap since the previous frame to the moved-time total
            # whenever the face is currently moved.
            dt = now - self._move_prev_t
            self._move_prev_t = now
            if moved:
                self._moved_time += dt

            # Count a new movement each time the face leaves the baseline.
            if moved and not self._move_away:
                self._move_count += 1
                self._move_away = True

            # Clear the away flag once back near the baseline.
            if not moved:
                self._move_away = False

            # Rule 1: too much total moved-time.
            if self._moved_time >= self.config.moved_time_limit_seconds:
                return MonitorResult(should_pause=True, reason="Too much movement")

            # Rule 2: too many separate movements.
            if self._move_count > self.config.move_count_limit:
                return MonitorResult(should_pause=True, reason="Too much movement")



        # ---- no problem this frame ----
        if reason is None:

            # Clear any problem timer, since the frame is fine.
            self._problem_started_at = None

            # Remember this face centre for the next jump check,
            # but only when a single face was present.
            if face_count == 1:
                self._last_centre = (centre_x, centre_y)

                # Set the baseline on the first good frame, so later
                # movement is measured from where the user actually began.
                if self._baseline_centre is None:
                    self._baseline_centre = (centre_x, centre_y)

            # Do not pause.
            return MonitorResult(should_pause=False)

        # ---- a problem is present ----

        # Start the problem timer if this is the first bad frame.
        if self._problem_started_at is None:
            self._problem_started_at = now

        # Work out how long the problem has lasted.
        problem_duration = now - self._problem_started_at

        # Pause only when the problem has lasted long enough.
        if problem_duration >= self.config.pause_after_seconds:
            return MonitorResult(should_pause=True, reason=reason)

        # Problem is still too brief to pause; keep recording for now.
        return MonitorResult(should_pause=False)

    # ------------------------------------------------------------------------

    def _find_problem(
        self,
        face_count: int,
        centre_x: float,
        centre_y: float,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> str | None:
        """Return a pause reason for this frame, or None when it is fine."""

        config = self.config

        # ---- (1) no face in the frame ----
        if face_count == 0:
            return "No face detected"

        # ---- (2) more than one face ----
        if face_count > 1:
            return "Another person appeared"
        
        # ---- face alignment: moved too far from the starting position ----
        # Movement is measured from the baseline centre captured when
        # recording began, so it adapts to wherever the user started. Left
        # and right are swapped in wording because the preview is mirrored.
        if self._baseline_centre is not None:

            # How far the face has moved from its starting point.
            dx = centre_x - self._baseline_centre[0]
            dy = centre_y - self._baseline_centre[1]

            # Collect any breach with how far past its limit it went, so the
            # strongest movement is the one reported.
            breaches = []

            # Face moved right in the frame; in the mirror the user moved left.
            if dx > config.max_move_horizontal:
                breaches.append(("You moved left", dx - config.max_move_horizontal))

            # Face moved left in the frame; in the mirror the user moved right.
            if dx < -config.max_move_horizontal:
                breaches.append(("You moved right", -dx - config.max_move_horizontal))

            # Face moved down.
            if dy > config.max_move_vertical:
                breaches.append(("You moved down", dy - config.max_move_vertical))

            # Face moved up.
            if dy < -config.max_move_vertical:
                breaches.append(("You moved up", -dy - config.max_move_vertical))

            # Report the strongest breach when any side is over its limit.
            if breaches:
                breaches.sort(key=lambda item: item[1], reverse=True)
                return breaches[0][0]
        

        # ---- Job A: head orientation (only when angles are known) ----
        if not math.isnan(yaw):

            # Turned too far left or right.
            if abs(yaw) > config.max_yaw_degrees:
                return "No face detected"

            # Tilted too far up or down.
            if abs(pitch) > config.max_pitch_degrees:
                return "You tilted your head"

            # Tilted sideways too far.
            if abs(roll) > config.max_roll_degrees:
                return "You tilted your head sideways"

        # ---- (6) sudden jump from the previous position ----
        # Checked last, so clearer position and orientation messages are
        # preferred over the generic jump message when both would apply.
        if self._last_centre is not None:

            # Distance the face centre moved since the previous frame.
            dx = centre_x - self._last_centre[0]
            dy = centre_y - self._last_centre[1]
            jump = math.sqrt(dx * dx + dy * dy)

            # A jump larger than the limit looks unnatural.
            if jump > config.max_jump_frac:
                return "Unexpected movement detected"

        # No problem found in this frame.
        return None