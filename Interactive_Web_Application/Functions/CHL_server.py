# ============================================================================
#  CHL_server.py — web server, routes and websocket
# ============================================================================
#  PURPOSE
#  -------
#  Serves the single-page frontend and holds the websocket endpoint that
#  later stages use to send camera frames to Python for analysis.
# ============================================================================

from __future__ import annotations

import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from Functions.CHL_config import APP_TITLE, FRONTEND_DIR

# base64 decodes the frames the browser sends as text.
import base64
 
# NumPy and OpenCV turn those bytes back into an image.
import numpy as np
import cv2
 
# The position checker and its config, plus the model path.
from Functions.CHL_position_checker import PositionChecker, PositionConfig
from Functions.CHL_config import LANDMARKER_MODEL

# The stability tracker that measures the continuous hold.
from Functions.CHL_stability import StabilityTracker

# The id maker that names each recording folder by date, time and counter.
from Functions.CHL_session_id import SessionIdMaker
 
# The capture session that saves frames and builds the video.
from Functions.CHL_capture import CaptureSession
 
# The capture folder, keep flag, and video frame rate from config.
from Functions.CHL_config import CAPTURE_DIR, KEEP_RECORDINGS, CAPTURE_FPS

# The recording monitor and its config.
from Functions.CHL_recording_monitor import RecordingMonitor, MonitorConfig

# Background model loading, the readiness wait, and per-video scoring.
from Functions.CHL_scoring import (
    start_background_load,
    wait_until_ready,
    score_recording,
    get_load_error,
)

# The challenge primitives and the random picker.
from Functions.CHL_challenge import pick_challenge, challenge_needs_hand

# The separate folder for the challenge-response recordings.
from Functions.CHL_config import CHALLENGE_CAPTURE_DIR

# The challenge validator and its outcome names.
from Functions.CHL_challenge_validator import (
    ChallengeValidator, PASS, FAIL, UNCLEAR,
)

# The identity checker (ArcFace) for the same-person check across stages.
from Functions.CHL_identity import IdentityChecker

# The per-session CSV logger for the study data.
from Functions.CHL_session_log import log_session



# ============================================================================
#  APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """
    Build the FastAPI application with its routes.

    Returns
    -------
    FastAPI
        The configured application.
    """
    app = FastAPI(title=APP_TITLE)

    # Serve stylesheet and script as static files under /static.
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )

    # Makes recording ids and resets the counter each new hour.
    id_maker = SessionIdMaker()

    @app.get("/")
    def index() -> FileResponse:
        """Serve the single page."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    
    checker = PositionChecker(LANDMARKER_MODEL, PositionConfig())

    # Build the challenge validator once; it loads its own models.
    validator = ChallengeValidator()

    # Build the identity checker once; it loads its own ArcFace model.
    identity_checker = IdentityChecker()

    # Begin loading the FAS model on a background thread. The server keeps
    # working while it loads, so it is usually ready by the time a recording
    # finishes.
    start_background_load()

    
    @app.websocket("/ws")
    async def frame_socket(websocket: WebSocket) -> None:
        """Receive frames, check face position, send back the message."""
 
        # Accept the browser's connection.
        await websocket.accept()

        # Track the position hold for this connection.
        stability = StabilityTracker(hold_seconds=3.0)

        # Holds the capture session once recording begins. None until then.
        capture: CaptureSession | None = None
 
        # How many frames still need saving to complete the recording.
        # Twenty seconds at the capture frame rate.
        frames_needed = int(20 * CAPTURE_FPS)

        # Watches the person during recording and decides when to pause.
        monitor = RecordingMonitor(MonitorConfig())

        # True once a violation has paused the recording. While paused, no
        # frames are checked or saved, so the recording freezes at the first
        # violation and its reason does not change.
        is_paused = False

        # True while checking the setup before the prompts, after the video
        # has been recorded and scored. Reuses the same position checks but
        # ends by signalling the prompts are ready, not by recording.
        prompt_setup = False

        # The stored passive verdict for this recording, used later together
        # with the challenge-response result. None until scoring runs.
        fas_verdict: str | None = None

        # The spoof and live probabilities from scoring, kept for the CSV row.
        fas_spoof_prob: float | None = None
        fas_live_prob: float | None = None

        # A list of per-attempt records for the session's CSV row. Each entry
        # is a dict with actions, outcomes, details, result, and swapped.
        # Empty until the first challenge attempt is validated.
        attempts = []

        # The action keys issued for the current attempt, stored so a failed
        # or swapped attempt can still record which actions were asked.
        current_actions = []

        # The researcher setup data for this session (participant, demographics,
        # condition, device), received before the recording. None until sent.
        session_setup = None

        # The session id of the passive recording, used to label the CSV row.
        current_session_id = None

        # The face embedding captured during the passive recording (Person A).
        # Used later to confirm the challenge is done by the same person.
        passive_embedding = None

        # The three primitives picked for this attempt. None until picked.
        challenge: list | None = None
 
        # Holds the challenge recording session once it begins. None until
        # the user presses Record Prompts.
        challenge_capture = None
 
        # How many frames make up the challenge recording. Fifteen seconds
        # at the capture frame rate.
        challenge_frames_needed = int(15 * CAPTURE_FPS)

        # How many challenge attempts have been used. Three are allowed.
        challenge_attempts = 0
 
        # The most that are allowed before a hard fail.
        MAX_CHALLENGE_ATTEMPTS = 3

        # Counts every challenge recording made this session, so each kept
        # recording gets its own numbered folder and none overwrite.
        challenge_record_count = 0

        # True right after a challenge recording starts, so the identity check
        # runs at the start of every attempt (including retries).
        pending_identity_check = False

        # Frames gathered at the start of an attempt for the identity check.
        # A few frames give a more reliable decision than a single frame.
        identity_frames = []

        # How many frames to gather before deciding identity. Small, so the
        # decision is made quickly, before the actions can be performed.
        IDENTITY_FRAMES_NEEDED = 3


        # Decide what happens after a challenge attempt, record it, and either
        # finish or issue a retry.
        async def _handle_challenge_outcome(ws, outcome, fas, swapped=False,
                                            detail=None):

            # These are updated here, so declare them nonlocal.
            nonlocal challenge, challenge_attempts, current_actions

            # ---- record this attempt in the attempts list ----
            if swapped:
                # An identity swap: logged plainly, counts as a failed attempt.
                attempts.append({
                    "actions": current_actions,
                    "outcomes": [], "details": [],
                    "result": "failed", "swapped": True,
                })
            else:
                passed = (outcome == PASS)
                attempts.append({
                    "actions": current_actions,
                    "outcomes": (detail or {}).get("outcomes", []),
                    "details": (detail or {}).get("details", []),
                    "result": "passed" if passed else "failed",
                    "swapped": False,
                })

            # ---- the challenge passed: final verdict, log, and return ----
            if (not swapped) and outcome == PASS:

                is_live = (fas == "LIVE")
                final_result = ("Liveness Passed" if is_live
                                else "Spoof: Passive failed, Challenge Passed")
                user_msg = "Liveness Passed" if is_live else "Not a real person"

                print("=" * 60)
                print("FINAL VERDICT")
                print(f"  Passive : {fas}   Challenge : PASSED")
                print(f"  Result  : {final_result}")
                print("=" * 60)

                await ws.send_json(
                    {"phase": "final_verdict", "live": is_live,
                     "fas": fas, "challenge": "PASS"}
                )
                log_session(
                    session_setup, current_session_id, attempts,
                    "PASSED", fas_verdict, fas_live_prob, fas_spoof_prob,
                    final_result, user_msg,
                    (session_setup or {}).get("device_browser", ""),
                )
                return

            # ---- failed or swapped: count the attempt against the limit ----
            challenge_attempts += 1

            # Out of attempts: a hard fail. Build the final result and message.
            if challenge_attempts >= MAX_CHALLENGE_ATTEMPTS:

                is_live = (fas == "LIVE")
                if is_live:
                    final_result = "Challenge Failed, Passive Passed"
                    user_msg = "Challenge Failed"
                else:
                    final_result = "Spoof: Passive failed, Challenge Failed"
                    user_msg = "Not a real person"

                print("=" * 60)
                print("FINAL VERDICT")
                print(f"  Passive : {fas}   Challenge : FAILED (attempts used)")
                print(f"  Result  : {final_result}")
                print("=" * 60)

                await ws.send_json(
                    {"phase": "final_verdict", "live": False,
                     "fas": fas, "challenge": "FAIL"}
                )
                log_session(
                    session_setup, current_session_id, attempts,
                    "FAILED", fas_verdict, fas_live_prob, fas_spoof_prob,
                    final_result, user_msg,
                    (session_setup or {}).get("device_browser", ""),
                )
                return

            # Attempts remain: pick a new, different challenge and ask again.
            challenge = pick_challenge(3)
            current_actions = [item["key"] for item in challenge]

            # How many attempts remain, for the browser legend.
            remaining = MAX_CHALLENGE_ATTEMPTS - challenge_attempts

            print(f"Challenge retry {challenge_attempts}/{MAX_CHALLENGE_ATTEMPTS} "
                  f"- new prompts issued ({remaining} remaining).")

            await ws.send_json(
                {
                    "phase": "challenge_retry",
                    "swapped": swapped,
                    "attempts_used": challenge_attempts,
                    "attempts_total": MAX_CHALLENGE_ATTEMPTS,
                    "attempts_remaining": remaining,
                    "prompts": [
                        {"key": item["key"], "text": item["text"],
                         "group": item["group"]}
                        for item in challenge
                    ],
                    "needs_hand": challenge_needs_hand(challenge),
                }
            )

 
        try:
            # Loop for as long as the browser keeps sending frames.
            while True:
 
                # Read the next message from the browser.
                data = await websocket.receive_text()

                # ---- setup data from the researcher screen ----
                # Sent as JSON before the session. Store it for the CSV row.
                if data.startswith("{"):
                    import json
                    try:
                        msg = json.loads(data)
                    except ValueError:
                        msg = None
                    if msg and msg.get("type") == "session_setup":
                        session_setup = msg.get("data", {})

                        # Generate the session id now, at Start Session, and
                        # append the participant code. The date and time make
                        # it unique even across refreshes; the hourly counter
                        # is left to behave exactly as before.
                        participant = session_setup.get("participant_code", "PXX")
                        current_session_id = f"{id_maker.new_id()}_{participant}"

                        print(f"Session setup received: {participant} "
                              f"| {session_setup.get('condition', '?')}")
                        print(f"Session id: {current_session_id}")
                        continue
 
                # ---- control signal: recording is starting ----
                if data == "start":
 
                    # Reuse the id generated at Start Session. Only build a
                    # fresh one as a fallback if setup was somehow skipped, so
                    # a recording always has an id.
                    if current_session_id is None:
                        current_session_id = id_maker.new_id()

                    # Use the session id to name the recording folder.
                    session_id = current_session_id

                    # Print the session id and participant for terminal tracing.
                    print(f"Session started: id={session_id} "
                          f"| participant={(session_setup or {}).get('participant_code', '?')}")
 
                    # Create the capture session and its folder.
                    capture = CaptureSession(CAPTURE_DIR, session_id, CAPTURE_FPS)

                    # Clear the monitor for this fresh recording.
                    monitor.reset()
 
                    # Tell the browser the id, for reference.
                    await websocket.send_json({"phase": "recording_started",
                                               "session_id": session_id})
 
                    # Wait for the next message.
                    continue
                
                # ---- control signal: begin setup checking for the prompts ----
                if data == "start_prompt_setup":

                    # Enter prompt-setup mode and clear the hold tracker so a
                    # fresh 3 second hold is required.
                    prompt_setup = True
                    stability.reset()

                    # Wait for the next frame.
                    continue

                
                # ---- control signal: start the challenge recording ----
                if data == "challenge_start":

                    # Number this challenge recording within the session.
                    challenge_record_count += 1

                    # Name the folder after the session id and attempt number,
                    # so every challenge recording is traceable to its
                    # participant and session, and attempts never overwrite.
                    base_id = current_session_id or id_maker.new_id()
                    challenge_folder = f"challenge_{base_id}_attempt{challenge_record_count}"
                    challenge_capture = CaptureSession(
                        CHALLENGE_CAPTURE_DIR, challenge_folder, CAPTURE_FPS
                    )

                    # Arm the identity check for the start of this attempt, so a
                    # swap on any attempt (including retries) is caught.
                    pending_identity_check = True
                    identity_frames = []
 
                    # Tell the browser recording has begun.
                    await websocket.send_json({"phase": "challenge_recording"})
 
                    # Wait for the next message.
                    continue
 
                # ---- control signal: finish the challenge early ----
                if data == "challenge_finish":
 
                    # Only act if a challenge recording exists.
                    if challenge_capture is not None:
 
                        # Assemble the recorded frames into a video, and keep
                        # the frames folder for the validator to read.
                        challenge_video = challenge_capture.assemble_video()
                        frames_dir = challenge_capture.frames_dir
 
                        # Tell the browser the response was recorded, so it
                        # shows the waiting screen while validation runs.
                        await websocket.send_json({"phase": "challenge_recorded"})
 
                        # The keys of the three expected primitives, in order.
                        expected = [item["key"] for item in challenge]
 
                        # Validate the recording against the expected order.
                        outcome, attempt_detail = validator.validate(frames_dir, expected)

                        # Print the challenge outcome for debugging.
                        print(f"Challenge outcome: {outcome}")
 
                        # Keep the recording for the study unless keeping is
                        # turned off, in which case delete it. The assembled
                        # video and frames stay on disk when kept.
                        if not KEEP_RECORDINGS:
                            challenge_capture.delete()
                        else:
                            print(f"Challenge recording kept: {challenge_video}")
                        challenge_capture = None
 
                        # Decide what happens next based on the outcome.
                        await _handle_challenge_outcome(
                            websocket, outcome, fas_verdict, detail=attempt_detail,
                        )
 
                    # Wait for the next message.
                    continue


                # ---- control signal: cancel and delete this recording ----
                if data == "stop":
 
                    # Delete the saved frames if a capture exists.
                    if capture is not None:
                        capture.delete()
                        capture = None
                    
                    # Clear the pause freeze so a fresh recording can run.
                    is_paused = False

                    # Also reset the monitor so the next recording starts clean.
                    monitor.reset()

                    # Clear the stored identity so a fresh run captures a new
                    # passive face rather than reusing the old one.
                    passive_embedding = None
 
                    # Wait for the next message.
                    continue

 
                # Strip the "data:image/jpeg;base64," prefix if present.
                if "," in data:
                    data = data.split(",", 1)[1]
 
                # Decode the base64 text into raw JPEG bytes.
                jpg_bytes = base64.b64decode(data)
 
                # Turn the bytes into a NumPy buffer.
                buffer = np.frombuffer(jpg_bytes, dtype=np.uint8)
 
                # Decode the JPEG buffer into a BGR image.
                frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
 
                # Skip if the frame did not decode.
                if frame_bgr is None:
                    continue
                               
                # ---- challenge recording: save frames until full ----
                if challenge_capture is not None:

                    # At the start of each attempt, gather a few frames and
                    # check identity, so a swap is caught fast, before the
                    # actions can be performed.
                    if pending_identity_check:

                        if passive_embedding is None:
                            # No passive face to compare against: skip the check.
                            print("IDENTITY CHECK skipped (no passive face captured).")
                            pending_identity_check = False
                            identity_frames = []
                        else:
                            identity_frames.append(frame_bgr)
                            if len(identity_frames) >= IDENTITY_FRAMES_NEEDED:
                                pending_identity_check = False
                                current_embedding = identity_checker.embed_from_frames(
                                    identity_frames
                                )
                                identity_frames = []
                                same, sim = identity_checker.is_same_person(
                                    passive_embedding, current_embedding
                                )
                                print("=" * 60)
                                print(f"IDENTITY CHECK (attempt {challenge_record_count})")
                                print(f"  Similarity   : {sim:.3f}")
                                print(f"  Same person? : {same}")
                                print("=" * 60)

                                if not same:
                                    # Swap: treat as a FAILED attempt. Discard
                                    # this recording (unless kept), tell the
                                    # browser, and route through the retry/final
                                    # handler so the attempt counts down.
                                    if not KEEP_RECORDINGS:
                                        challenge_capture.delete()
                                    else:
                                        challenge_capture.assemble_video()
                                    challenge_capture = None
                                    await websocket.send_json(
                                        {"phase": "identity_swapped"}
                                    )
                                    await _handle_challenge_outcome(
                                        websocket, FAIL, fas_verdict,
                                        swapped=True,
                                    )
                                    continue

                    # Save this frame to the challenge folder.
                    if challenge_capture is not None:
                        challenge_capture.add_frame(frame_bgr)

                    # When fifteen seconds of frames are gathered, finish
                    # automatically even if the user did not press Finish.
                    if challenge_capture.frame_count >= challenge_frames_needed:

                        # Assemble and keep the frames folder.
                        challenge_video = challenge_capture.assemble_video()
                        frames_dir = challenge_capture.frames_dir

                        # Show the waiting screen.
                        await websocket.send_json({"phase": "challenge_recorded"})

                        # Validate against the expected order.
                        expected = [item["key"] for item in challenge]
                        outcome, attempt_detail = validator.validate(frames_dir, expected)
                        print(f"Challenge outcome: {outcome}")

                        # Keep the recording for the study unless keeping is
                        # turned off, in which case delete it.
                        if not KEEP_RECORDINGS:
                            challenge_capture.delete()
                        else:
                            print(f"Challenge recording kept: {challenge_video}")
                        challenge_capture = None

                        # Branch on the outcome.
                        await _handle_challenge_outcome(
                            websocket, outcome, fas_verdict,detail=attempt_detail,
                        )

                    # Do not run position checks on challenge frames.
                    continue


                # While paused, ignore every frame so the recording stays
                # frozen at the first violation until the user resumes or
                # starts new. No checks run, no frames are saved, no new
                # reasons are sent.
                if is_paused:
                    continue

                # ---- prompt-setup mode: re-check position before prompts ----
                # This mirrors the entry gate exactly: the same position
                # check and the same 3 second stability hold. A violation at
                # any point keeps the user on the coaching screen, and the
                # hold countdown resets. Only a full, held position leads to
                # the prompts.
                if prompt_setup:

                    # Run the same position check as the entry gate.
                    check = checker.check(frame_bgr)

                    # Update the hold tracker with this frame's result. This
                    # resets on any violation, exactly like before recording.
                    hold = stability.update(check.ok)

                    # When the position has been held long enough, tell the
                    # browser to show the "Showing your prompt" countdown and
                    # leave prompt-setup mode.
                    if hold.phase == "start_recording":
 
                        # Leave prompt-setup mode.
                        prompt_setup = False

                        # The identity check now runs at the start of every
                        # challenge attempt (see the challenge-recording block),
                        # so a swap during any attempt is caught.

                        # Pick three random primitives for this attempt.
                        challenge = pick_challenge(3)

                        # Remember the action keys for this attempt's log entry.
                        current_actions = [item["key"] for item in challenge]
 
                        # Send the challenge to the browser: the ready signal
                        # plus the three prompts and whether a hand is needed.
                        await websocket.send_json(
                            {
                                "phase": "prompt_ready",
                                "prompts": [
                                    {"key": item["key"], "text": item["text"],
                                     "group": item["group"]}
                                    for item in challenge
                                ],
                                "needs_hand": challenge_needs_hand(challenge),
                            }
                        )
                        continue

                    # While checking, send the coaching message so the user
                    # is guided into position, same as the entry gate.
                    if hold.phase == "checking":
                        await websocket.send_json(
                            {"phase": "prompt_checking", "message": check.message}
                        )
                        continue

                    # While holding, send the countdown so the user holds
                    # still. This is the server's held-position countdown, so
                    # moving resets it.
                    if hold.phase == "holding":
                        await websocket.send_json(
                            {"phase": "prompt_holding", "countdown": hold.countdown}
                        )
                        continue

                    # Nothing else to do for this frame.
                    continue

                # Run the position check on this frame.
                result = checker.check(frame_bgr)
 
                # Update the hold tracker with this frame's position result.
                state = stability.update(result.ok)

                # While recording, save each frame and track progress.
                if capture is not None:
 
                    # Read the values the monitor needs from the check.
                    # face_count : how many faces were found
                    # cx, cy     : face centre in the 0..1 range
                    # yaw/pitch/roll : head angles in degrees (NaN if unknown)
                    face_count = result.face_count
                    cx = result.centre_x
                    cy = result.centre_y
                    yaw = result.yaw
                    pitch = result.pitch
                    roll = result.roll
 
                    # Ask the monitor whether to pause on this frame.
                    watch = monitor.update(
                        face_count, cx, cy, yaw, pitch, roll,
                    )
 
                    # When a pause is needed, tell the browser and stop
                    # saving frames. The frames so far are kept, not deleted.
                    if watch.should_pause:

                        # Freeze the recording so no further frames are
                        # checked and the reason cannot change.
                        is_paused = True
 
                        # Send the pause signal and the reason.
                        await websocket.send_json(
                            {"phase": "recording_paused", "reason": watch.reason}
                        )
 
                        # Keep the capture object so recording can resume
                        # later, but do not save this frame.
                        continue
                    
                    # Save this frame to the recording folder.
                    capture.add_frame(frame_bgr)
 
                    # Check whether enough frames have been captured.
                    if capture.frame_count >= frames_needed:
 
                        # Join the frames into the final video file.
                        video_path = capture.assemble_video()
 
                        # Tell the browser recording is done (shows the
                        # centered "Recording complete" message).
                        await websocket.send_json(
                            {"phase": "recording_complete"}
                        )
 
                        # Tell the browser scoring is starting (shows
                        # "Checking video...").
                        await websocket.send_json(
                            {"phase": "checking_video"}
                        )
 
                        # Wait until the background model load has finished.
                        # Usually already done thanks to the recording time.
                        ready = wait_until_ready(timeout=60.0)
 
                        if not ready:
 
                            # Loading failed or timed out. Report the error
                            # and do not proceed to the next stage.
                            await websocket.send_json(
                                {"phase": "score_error",
                                 "detail": get_load_error() or "model not ready"}
                            )
 
                            # Keep the frames; the user will start new.
                            capture = None
                            continue
 
                        # Capture the passive-stage face identity from the
                        # saved frames, for the later same-person check.
                        try:
                            import glob
                            frame_paths = sorted(
                                glob.glob(str(capture.frames_dir / "frame_*.jpg"))
                            )
                            # Use a few frames spread across the recording so a
                            # blurred or turned frame does not defeat it.
                            sample_paths = frame_paths[:: max(1, len(frame_paths) // 5)]
                            sample_frames = [cv2.imread(p) for p in sample_paths]
                            passive_embedding = identity_checker.embed_from_frames(
                                [f for f in sample_frames if f is not None]
                            )
                            if passive_embedding is None:
                                print("Identity: no face captured in passive stage.")
                            else:
                                print("Identity: passive-stage face captured.")
                        except Exception as id_err:
                            print(f"Identity capture failed: {id_err}")
                            passive_embedding = None

                        # Score the video. Any failure is caught, printed to
                        # the terminal, and reported to the browser.
                        try:
                            outcome = score_recording(video_path)
                            fas_verdict = outcome["verdict"]  # 'LIVE' or 'SPOOF'

                            # Keep the spoof and live probabilities for the CSV.
                            fas_spoof_prob = outcome["spoof_prob"]
                            fas_live_prob = 1.0 - outcome["spoof_prob"]
 
                            # Log the verdict to the terminal for reference.
                            print(f"FAS verdict: {fas_verdict} "
                                  f"(spoof_prob={outcome['spoof_prob']:.3f})")
 
                            # Tell the browser scoring succeeded, so it can
                            # move on to the liveness prompts. The verdict is
                            # kept here on the server, not shown to the user.
                            await websocket.send_json(
                                {"phase": "score_done"}
                            )
 
                        except Exception as error:
 
                            # Print the exact error to the terminal.
                            import traceback
                            print("SCORING FAILED:")
                            traceback.print_exc()
 
                            # Tell the browser to show the error screen.
                            await websocket.send_json(
                                {"phase": "score_error",
                                 "detail": str(error)}
                            )
 
                        # Delete the folder only when keeping is turned off.
                        if not KEEP_RECORDINGS:
                            capture.delete()
 
                        # Clear the capture so no more frames are saved.
                        capture = None
 
                        # Skip the position reply for this final frame.
                        continue
 
                # Send the position result together with the hold phase and
                # the countdown number, so the browser can guide the user.
                await websocket.send_json(
                    {
                        "ok": result.ok,          # position correct or not
                        "message": result.message, # instruction text
                        "phase": state.phase,      # checking / holding / start_recording
                        "countdown": state.countdown,  # 3, 2, 1 while holding
                    }
                )
 
        # The browser closed the connection; nothing to clean up.
        except WebSocketDisconnect:
            pass

    return app


# ============================================================================
#  LAUNCH
# ============================================================================

def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """
    Start the web server.

    Parameters
    ----------
    host : str
        Interface to bind. The loopback default keeps the server reachable
        only from this machine.
    port : int
        Port to listen on.
    open_browser : bool
        Whether to open a browser tab once the server is starting.

    Notes
    -----
    The browser is opened from a short-lived timer thread so that it fires
    just after the server begins listening, rather than blocking startup.
    """
    if open_browser:
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{host}:{port}")
        ).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")