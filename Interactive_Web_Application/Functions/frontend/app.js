// ==========================================================================
//  app.js — camera startup, position-check streaming, and controls
// ==========================================================================
//
//  Flow:
//    1. Start the camera on page load.
//    2. When Start Recording is pressed, hide the button, show the
//       "Checking setup..." legend, and start sending frames to Python.
//    3. Python replies with an instruction; show it in the middle of the
//       screen until the position is correct.
// ==========================================================================

(function () {
    "use strict";

    // ---- grab the elements once -----------------------------------------

    // The live camera video element.
    const video = document.getElementById("camera");

    // The permission notice shown before the camera starts.
    const permission = document.getElementById("permission");

    // The text inside the permission notice.
    const permissionText = document.getElementById("permission-text");

    // The circular record button and its label.
    const recordButton = document.getElementById("record");
    const controlBar = document.getElementById("control-bar");

    // The middle instruction pill and its text.
    const coach = document.getElementById("coach");
    const coachText = document.getElementById("coach-text");

    // The bottom "Checking setup..." legend.
    const legend = document.getElementById("legend");

    // The message pill shown after START, and its text.
    const message = document.getElementById("message");
    const messageText = document.getElementById("message-text");

    // The recording area, its progress ring, and the countdown legend.
    const recArea = document.getElementById("rec-area");
    const recProgress = document.getElementById("rec-progress");
    const recCountdown = document.getElementById("rec-countdown");
    
    // Total recording length in seconds.
    const RECORD_SECONDS = 20;
    
    // The full circle length of the ring, matching stroke-dasharray in CSS.
    // Circumference = 2 * pi * r = 2 * 3.1416 * 46 = about 289.
    const RING_LENGTH = 289;
    
    // Holds the running timer so it can be stopped later.
    let recTimer = null;
    
    // Seconds elapsed so far in the current recording.
    let recElapsed = 0;

    // The canvas that shows the frozen frame while paused.
    const freeze = document.getElementById("freeze");

    // The after-recording centre panel and its parts. 
    const afterRecord = document.getElementById("after-record"); 
    const arSub = document.getElementById("ar-sub"); 
    const arSub2 = document.getElementById("ar-sub2"); 
    const startPromptsBtn = document.getElementById("start-prompts-btn"); 
    const arStartNewBtn = document.getElementById("ar-startnew-btn");

    // The top-middle "Prompts Check" legend.
    const promptCheckLegend = document.getElementById("prompt-check-legend");

    // The challenge prompt panel and its parts.
    const promptPanel = document.getElementById("prompt-panel");
    const recordPromptsBtn = document.getElementById("record-prompts-btn");
    const promptText = document.getElementById("prompt-text");

    // The face guide oval.
    const faceOval = document.getElementById("face-oval");

    // The researcher setup screen, its error banner, and Start button.
    const setupScreen = document.getElementById("setup-screen");
    const setupBanner = document.getElementById("setup-error-banner");
    const startSessionBtn = document.getElementById("start-session-btn");

    // The identity-swap warning and the attempts-remaining legend.
    const swapWarning = document.getElementById("swap-warning");
    const attemptsLegend = document.getElementById("attempts-legend");
    
    

    // ---- state ----------------------------------------------------------

    // The websocket connection to Python.
    let socket = null;

    // True once position checking has started.
    let checking = false;

    // A hidden canvas used to grab JPEG frames from the video.
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    // How many frames to send per second.
    const FRAMES_PER_SECOND = 8;

    // True once recording has begun, so the timer is not started twice.
    let recordingStarted = false;

    // True once the challenge recording has begun.
    let challengeRecording = false;

    // Holds the next set of prompts after a failed attempt, shown when the
    // user taps Try Again.
    let pendingPrompts = null;
    
    // Holds the confirmed setup data for this session, attached to the CSV
    // row when the session completes. Null until Start Session succeeds.
    let sessionSetup = null;

    // True once Start Session has been pressed once, so fields re-validate live.
    let setupSubmitted = false;

    // True when the last attempt was an identity swap, so the next failed
    // screen shows the orange warning.
    let pendingSwapWarning = false;


    // ---- camera startup -------------------------------------------------

    // Ask the browser for the camera and show it in the video element.
    async function startCamera() {
        try {
            // Request an upright camera stream.
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 720 },
                    height: { ideal: 1280 }
                },
                audio: false
            });

            // Feed the stream into the video element.
            video.srcObject = stream;

            // Hide the permission notice now that the camera is live.
            permission.classList.add("hidden");
        } catch (error) {
            // Show a clear message if the camera is blocked or missing.
            permissionText.textContent =
                "Camera unavailable. Please allow camera access and reload.";
            console.error("getUserMedia failed:", error);
        }
    }

    // ---- websocket ------------------------------------------------------

    // Open the connection to Python and handle replies.
    function openSocket() {
        // Choose ws or wss to match the page protocol.
        const protocol = location.protocol === "https:" ? "wss" : "ws";

        // Connect to the /ws endpoint on the same host.
        socket = new WebSocket(`${protocol}://${location.host}/ws`);

        // Log when the connection opens.
        socket.onopen = () => console.log("frame socket open");

        // Handle each reply from Python.
        socket.onmessage = (event) => {
 
            // Parse the JSON reply.
            const reply = JSON.parse(event.data);

            // ---- server reports the recording has paused ----
            if (reply.phase === "recording_paused") {
 
                // Match the freeze canvas size to the video.
                freeze.width = video.videoWidth;
                freeze.height = video.videoHeight;

                // Draw the current video frame onto the freeze canvas.
                freeze.getContext("2d").drawImage(
                    video, 0, 0, freeze.width, freeze.height
                );

                // Show the frozen frame over the live camera.
                freeze.classList.remove("hidden");

                // Put the dynamic reason into the warning message.
                document.getElementById("pause-reason").textContent = reply.reason;

                // Show the pause screen with its buttons.
                document.getElementById("pause-screen").classList.remove("hidden");

                // Grey out and freeze the record button and timer.
                recArea.classList.add("paused");

                // Stop the timer sweep so the ring holds where it paused.
                if (recTimer) {
                    clearInterval(recTimer);
                    recTimer = null;
                }

                // Hide the face guide oval while paused.
                faceOval.classList.add("hidden");

                return;
            }

            // ---- server confirms recording has begun ----
            if (reply.phase === "recording_started") {
 
                // Log the recording id for reference.
                console.log("Recording started, id:", reply.session_id);
 
                // Nothing else to do; frames are already being sent.
                return;
            }
 
            // ---- recording finished: show the centre panel ----
            if (reply.phase === "recording_complete") {

                // The clock already showed the panel; nothing to do here.
                return;
            }

            // ---- scoring is running ----
            if (reply.phase === "checking_video") {

                // Keep the "Please wait" line and spinner already showing.
                return;
            }

            // ---- scoring succeeded: run the liveness intro ----
            if (reply.phase === "score_done") {

                // Start the short liveness intro countdown, then show the
                // explanation and the Start Prompts button.
                startLivenessIntro();
                return;
            }

            // ---- scoring failed: show the error and Start New ----
            if (reply.phase === "score_error") {

                // Log the exact error for reference.
                console.log("Scoring error:", reply.detail);

                // Replace the heading with the error and offer Start New.
                document.getElementById("ar-title").textContent =
                    "Error. Please record again";
                arSub.textContent = "";
                arSub2.classList.add("hidden");
                startPromptsBtn.classList.add("hidden");
                arStartNewBtn.classList.remove("hidden");
                return;
            }


             // ---- prompt setup: position not yet correct ----
             if (reply.phase === "prompt_checking") {
                messageText.textContent = reply.message;
                // Not correct yet: white oval.
                faceOval.classList.remove("ok");
                return;
            }

            // ---- prompt setup: holding, show the server countdown ----
            // This countdown comes from the server's stability hold, so if
            // the user moves it resets, exactly like the entry gate.
            if (reply.phase === "prompt_holding") {
                messageText.textContent =
                    "Please hold this position. Showing your prompt... " + reply.countdown;
                
                // Position is correct: turn the oval green.
                faceOval.classList.add("ok");
                return;
            }

            // ---- prompt setup complete: show the challenge prompts ----
            if (reply.phase === "prompt_ready") {

                // Hide the setup UI.
                promptCheckLegend.classList.add("hidden");
                legend.classList.add("hidden");
                messageText.textContent = "";
                message.classList.add("hidden");

                // Hide the face guide oval; the prompt panel is now shown and
                // the user is about to perform actions, not hold still.
                faceOval.classList.add("hidden");
                faceOval.classList.remove("ok");

                // Stop sending setup frames for now.
                checking = false;

                // Build and show the challenge prompt panel.
                showPromptPanel(reply.prompts);
                return;
            }

            // ---- challenge recording has begun ----
            if (reply.phase === "challenge_recording") {

                // Nothing to show; frames are already being sent.
                console.log("Challenge recording started");
                return;
            }

            // ---- challenge recording finished ----
            if (reply.phase === "challenge_recorded") {
                console.log("Challenge recorded:", reply.video);
                challengeRecording = false;
                checking = false;

                // Stop the countdown and reset the button look.
                stopChallengeCountdown();
                recordPromptsBtn.classList.remove("recording");

                promptPanel.classList.add("hidden");

                // Show the "Live Response Recorded" waiting screen, reusing
                // the after-record panel with a spinner. Detection comes in
                // later sub-tasks.
                afterRecord.classList.remove("hidden");
                document.getElementById("ar-title").classList.remove("hidden");
                document.getElementById("ar-title").textContent =
                    "Live Response Recorded";
                arSub.classList.remove("hidden");
                arSub.textContent = "Please wait";
                arSub.classList.add("with-spinner");
                arSub2.classList.add("hidden");
                startPromptsBtn.classList.add("hidden");
                arStartNewBtn.classList.add("hidden");
                
                // Hide the face guide oval on the result screen.
                faceOval.classList.add("hidden");

                return;
            }


            // ---- challenge failed, unclear, or identity swap: Failed screen ----
            if (reply.phase === "challenge_retry") {

                // Hide the waiting panel.
                afterRecord.classList.add("hidden");
                document.getElementById("ar-title").classList.remove("hidden");
                arSub.classList.remove("with-spinner");

                // Keep the fresh prompts aside until the user taps Try Again.
                // Nothing is said about which action failed.
                pendingPrompts = reply.prompts;

                // Show the orange warning ONLY when this specific retry
                // followed an identity swap. Reset the flag immediately so it
                // can never carry over to a later normal failure.
                const wasSwap = (reply.swapped === true);
                pendingSwapWarning = false;

                if (wasSwap) {
                    swapWarning.classList.remove("hidden");
                    document.getElementById("failed-text").textContent =
                        "Failed. Please be the same person and record the prompts.";
                } else {
                    swapWarning.classList.add("hidden");
                    document.getElementById("failed-text").textContent =
                        "Failed. Please record the prompts properly.";
                }

                // Show the "N / 3 attempts remaining" legend.
                if (reply.attempts_remaining !== undefined &&
                    reply.attempts_remaining !== null) {
                    attemptsLegend.textContent =
                        reply.attempts_remaining + " / " +
                        reply.attempts_total + " attempts remaining";
                    attemptsLegend.classList.remove("hidden");
                } else {
                    attemptsLegend.classList.add("hidden");
                }

                // Show the Failed screen with the Try Again button.
                document.getElementById("failed-screen").classList.remove("hidden");
                return;
            }

            // ---- the final verdict has been decided ----
            if (reply.phase === "final_verdict") {

                // Hide the waiting panel.
                afterRecord.classList.add("hidden");

                // Find the verdict parts.
                const screen = document.getElementById("verdict-screen");
                const icon = document.getElementById("verdict-icon");
                const text = document.getElementById("verdict-text");

                // The three icon shapes, reused below.
                const tickSvg =
                    '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" ' +
                    'stroke-width="3" stroke-linecap="round" ' +
                    'stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
                const crossSvg =
                    '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" ' +
                    'stroke-width="3" stroke-linecap="round" ' +
                    'stroke-linejoin="round"><line x1="18" y1="6" x2="6" ' +
                    'y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
                const bangSvg =
                    '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" ' +
                    'stroke-width="3" stroke-linecap="round" ' +
                    'stroke-linejoin="round"><line x1="12" y1="6" x2="12" ' +
                    'y2="14"/><line x1="12" y1="18" x2="12" y2="18"/></svg>';

                // Pick the screen from the FAS verdict and challenge result.
                if (reply.live) {
                    // LIVE + PASS: genuine live person.
                    icon.className = "pass";
                    icon.innerHTML = tickSvg;
                    text.textContent = "Liveness Passed";
                } else if (reply.fas === "LIVE" && reply.challenge === "FAIL") {
                    // LIVE + challenge failed: seems real, could not complete.
                    icon.className = "warn";
                    icon.innerHTML = bangSvg;
                    text.textContent = "Challenge Failed";
                } else {
                    // Any spoof outcome (SPOOF + PASS, or SPOOF + FAIL):
                    // a real spoof rejection.
                    icon.className = "fail";
                    icon.innerHTML = crossSvg;
                    text.textContent = "Not a real person";
                }

                // Show the verdict screen.
                screen.classList.remove("hidden");

                // Hide the face guide oval on the result screen.
                faceOval.classList.add("hidden");

                return;
            }

            // ---- identity swapped on this attempt ----
            // The server treats this as a failed attempt and follows it with a
            // challenge_retry (or a final_verdict if attempts are used up).
            // Just flag that the next failed screen shows the swap warning,
            // and stop the challenge recording UI.
            if (reply.phase === "identity_swapped") {
                pendingSwapWarning = true;
                challengeRecording = false;
                checking = false;
                stopChallengeCountdown();
                recordPromptsBtn.classList.remove("recording");
                promptPanel.classList.add("hidden");
                return;
            }

 
            // Ignore replies once recording has already started.
            if (recordingStarted) {
                return;
            }
 
            // ---- position not yet correct: show the correction prompt ----
            if (reply.phase === "checking") {
 
                // Put the instruction in the middle pill.
                messageText.textContent = reply.message;
                
                // Position not correct yet: keep the oval white.
                faceOval.classList.remove("ok");

                return;
            }
 
            // ---- position correct and holding: show the hold countdown ----
            if (reply.phase === "holding") {
 
                // Show the hold request and the countdown number below it.
                messageText.textContent =
                    "Please hold this position. A 20 second video will be taken.  " +
                    "Starting... " + reply.countdown;
                // Position is correct: turn the oval green.
                faceOval.classList.add("ok");
                return;
            }
 
            // ---- hold complete: start the recording ----
            if (reply.phase === "start_recording") {
 
                // Mark recording as started so this runs only once.
                recordingStarted = true;
 
                // Clear the middle message.
                messageText.textContent = "";

                // Hide the message element so its empty pill does not show.
                message.classList.add("hidden");
 
                // Hide the bottom legend.
                legend.classList.add("hidden");
 
                // Begin the recording button and its timer.
                startRecordingTimer();

                // Tell the server to begin saving frames.
                socket.send("start");

                return;
            }

        };

        // Log errors and closure.
        socket.onerror = (e) => console.error("frame socket error", e);
        socket.onclose = () => console.log("frame socket closed");
    }

    // ---- sending frames -------------------------------------------------

    // Grab the current video frame and send it to Python as JPEG text.
    function sendFrame() {
        // Stop if checking has ended or the socket is not ready.
        if (!checking || !socket || socket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Skip until the video has real dimensions.
        if (!video.videoWidth || !video.videoHeight) {
            return;
        }

        // Match the canvas size to the video frame.
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw the current video frame onto the canvas.
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Encode the canvas as a JPEG data URL (quality 0.6 keeps it light).
        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);

        // Send the encoded frame to Python.
        socket.send(dataUrl);
    }

    // ---- coaching message -----------------------------------------------

    // Show an instruction in the middle of the screen.
    function showCoach(message) {
        // Put the text into the pill.
        coachText.textContent = message;

        // Make the pill visible.
        coach.classList.remove("hidden");
    }

    // ---- record button --------------------------------------------------

    // Start position checking when the record button is pressed.
    // Handle the START press.
    function onRecord() {
 
        // Switch the button to its recording look:
        // START disappears and the red fills the circle.
        recordButton.classList.add("recording");
    
        // The button should vanish completely, so hide it right after.
        // (The class change is not strictly needed once hidden, but it keeps
        //  the state correct if the button is shown again later.)
        recordButton.style.display = "none";
    
        // Show a readable message low on the screen, above the button area.
        legend.textContent = "Checking your position…";
        legend.classList.remove("hidden");

        // Middle pill: make it visible now; live prompts fill it shortly.
        message.classList.remove("hidden");

        // Mark that checking has started, so sendFrame is allowed to run.
        checking = true;

        // Start sending frames to Python at the chosen rate.
        setInterval(sendFrame, 1000 / FRAMES_PER_SECOND);

        // Show the face guide oval while positioning and recording.
        faceOval.classList.remove("hidden");
    }


    // Start the visual recording timer: sweep the ring and count down.
    function startRecordingTimer() {
    
        // Show the recording area.
        recArea.classList.remove("hidden");
    
        // Reset elapsed time to the start.
        recElapsed = 0;
    
        // Update the ring and legend every tenth of a second for smoothness.
        const tickSeconds = 0.1;
    
        // Clear any timer left from before, so two never run at once.
        if (recTimer) {
            clearInterval(recTimer);
        }
    
        // Run the tick repeatedly.
        recTimer = setInterval(() => {
    
            // Add the tick length to the elapsed time.
            recElapsed += tickSeconds;
    
            // Work out how far through the recording the timer is (0 to 1).
            const fraction = Math.min(recElapsed / RECORD_SECONDS, 1);
    
            // Move the ring: offset shrinks from full length to zero as the
            // sweep fills. A smaller offset shows more of the ring.
            recProgress.style.strokeDashoffset = RING_LENGTH * (1 - fraction);
    
            // Seconds remaining, rounded up so it reads 20 down to 0.
            const remaining = Math.ceil(RECORD_SECONDS - recElapsed);
    
            // Show the remaining seconds in the legend, never below zero.
            recCountdown.textContent = Math.max(remaining, 0) + "s";
    
            // Stop when the full time has passed.
            if (recElapsed >= RECORD_SECONDS) {
                clearInterval(recTimer);
                recTimer = null;

                // Hide the timer. The centre panel is shown when the server
                // confirms the recording is complete.
                recArea.classList.add("hidden");

                // Hide the face guide oval once recording ends, reset to white.
                faceOval.classList.add("hidden");
                faceOval.classList.remove("ok");

                // Show the after-recording panel at once, so the user is not
                // left waiting. Heading plus a "Please wait" line and a
                // spinner, until scoring finishes.
                afterRecord.classList.remove("hidden");
                arSub.textContent = "Please wait";
                arSub.classList.add("with-spinner");
                arSub2.classList.add("hidden");
                startPromptsBtn.classList.add("hidden");
                arStartNewBtn.classList.add("hidden");

            }
        }, tickSeconds * 1000);
    }

    // TEMP: expose for console testing. Remove later.
    //window.startRecordingTimer = startRecordingTimer;

    
    // Handle the Start New button: delete this recording and restart.
    function onStartNew() {

        // Tell the server to delete the saved frames for this recording.
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send("stop");
        }

        // Reset the whole page back to the start.
        resetToStart();
    }

    // Return the interface to its first state, ready for a new recording.
    function resetToStart() {

        // Hide the pause screen and its buttons.
        document.getElementById("pause-screen").classList.add("hidden");

        // Hide the frozen frame so the live camera shows again.
        freeze.classList.add("hidden");

        // Hide the recording area and remove the greyed look.
        recArea.classList.add("hidden");
        recArea.classList.remove("paused");

        // Clear any running timer.
        if (recTimer) {
            clearInterval(recTimer);
            recTimer = null;
        }

        // Reset the timer ring and countdown to their starting look.
        recProgress.style.strokeDashoffset = RING_LENGTH;
        recCountdown.textContent = RECORD_SECONDS + "s";

        // Hide the middle message and the bottom legend.
        message.classList.add("hidden");
        legend.classList.add("hidden");

        // Stop sending frames and clear the checking and recording flags.
        checking = false;
        recordingStarted = false;

        // Bring back the START button.
        recordButton.classList.remove("recording");
        recordButton.style.display = "";

        // Hide the after-recording panel and restore its heading.
        document.getElementById("after-record").classList.add("hidden");
        document.getElementById("ar-title").textContent = "Recording complete";
        document.getElementById("ar-title").classList.remove("hidden");

        // Restore the sub line so it can show "Please wait" next time.
        arSub.classList.remove("hidden");
        arSub.classList.remove("with-spinner");

        // Hide the final verdict screen for a fresh run.
        document.getElementById("verdict-screen").classList.add("hidden");

        // Hide the failed screen for a fresh run.
        document.getElementById("failed-screen").classList.add("hidden");

        // Hide the swap warning and attempts legend for a fresh run.
        swapWarning.classList.add("hidden");
        attemptsLegend.classList.add("hidden");
        pendingSwapWarning = false;

        // Hide the face guide oval for a fresh run, reset to white.
        faceOval.classList.add("hidden");
        faceOval.classList.remove("ok");
    }

    
    // Run the "Starting liveness component" countdown, then show the
    // explanation and the Start Prompts button.
    function startLivenessIntro() {
        
        // Remove the waiting spinner now that scoring is done.
        arSub.classList.remove("with-spinner");

        // Start the count at three.
        let count = 3;

        // Show the first number at once.
        arSub.textContent = "Starting liveness component… " + count;

        // Step down once per second.
        const timer = setInterval(() => {

            // Move to the next number.
            count -= 1;

            // While numbers remain, show the next one.
            if (count > 0) {
                arSub.textContent = "Starting liveness component… " + count;
                return;
            }

            // Countdown finished: stop the timer.
            clearInterval(timer);

            // Hide the "Recording complete" heading now.
            document.getElementById("ar-title").classList.add("hidden");

            // Hide the empty sub line so its box does not linger.
            arSub.textContent = "";
            arSub.classList.add("hidden");

            // Show the explanation and the Start Prompts button.
            arSub2.textContent =
                "You will now be asked to do some prompts, do them as requested.";
            arSub2.classList.remove("hidden");
            startPromptsBtn.classList.remove("hidden");
        }, 1000);
    }


    // Build and show the challenge prompt panel from the three prompts.
    function showPromptPanel(prompts) {

        // Hide the face guide oval whenever the prompt panel is shown (both
        // the first prompts and any Try Again retry), since the user is
        // about to perform actions, not hold still.
        faceOval.classList.add("hidden");
        faceOval.classList.remove("ok");

        // Join the three prompt texts into one line, lower-cased after the
        // first so it reads as a sentence.
        const line = prompts.map((p) => p.text).join(", ");
        promptText.textContent = line;

        // If any prompt is a hand gesture, remind the user to show the hand
        // flat to the camera, since the detection needs that orientation.
        const hasHand = prompts.some((p) => p.group === "hand");
        const instruction = document.getElementById("prompt-instruction");
        if (hasHand) {
            instruction.textContent =
                "Please click Record Prompts and perform the three actions " +
                "as a single video (15 seconds). For hand signs, show your " +
                "hand flat and facing the camera.";
        } else {
            instruction.textContent =
                "Please click Record Prompts and perform the three actions " +
                "as a single video (total time is only 15 seconds).";
        }

        // Reset the record button to its starting label.
        recordPromptsBtn.textContent = "Record Prompts";

        // Show the panel.
        promptPanel.classList.remove("hidden");
    }

    // Handle the Record / Finish button on the prompt panel.
    function onRecordPrompts() {

        // If not recording yet, start the challenge recording.
        if (!challengeRecording) {

            // Mark recording on and switch the button label and colour.
            challengeRecording = true;
            recordPromptsBtn.textContent = "Finish Recording Prompts";
            recordPromptsBtn.classList.add("recording");

            // Make sure frames are being sent to the server.
            checking = true;

            // Replace the instruction with a 15 second countdown timer.
            startChallengeCountdown();

            // Hide the oval during the challenge, since the user now moves
            // their head and hands to perform the actions.
            faceOval.classList.add("hidden");

            // Tell the server to begin the challenge recording.
            socket.send("challenge_start");
            return;
        }

        // Already recording: finish it now.
        challengeRecording = false;

        // Stop the countdown and reset the button look.
        stopChallengeCountdown();
        recordPromptsBtn.classList.remove("recording");

        socket.send("challenge_finish");
    }


    // ---- wiring ---------------------------------------------------------


    // Mark a field as bad or clear it; return whether it is valid.
    function setFieldError(fieldEl, ok) {
        if (ok) {
            fieldEl.classList.remove("bad");
        } else {
            fieldEl.classList.add("bad");
        }
        return ok;
    }

    // Check every field, show the errors, and return the error count.
    function validateSetup() {

        // Grab the six inputs.
        const code = document.getElementById("su-code");
        const condition = document.getElementById("su-condition");
        const age = document.getElementById("su-age");
        const gender = document.getElementById("su-gender");
        const ethnicity = document.getElementById("su-ethnicity");
        const aids = document.getElementById("su-aids");

        // Count errors as each field is checked.
        let errors = 0;

        // Participant code must be P followed by a number of 1 or more, so
        // P01, P02, P100 pass, but P00, P0 and other formats fail.
        const codeValue = code.value.trim();
        const codeOk = /^P\d+$/.test(codeValue) && parseInt(codeValue.slice(1), 10) >= 1;
        if (!setFieldError(code.parentElement, codeOk)) errors++;

        // Each dropdown must have a non-empty choice.
        if (!setFieldError(condition.parentElement, condition.value !== "")) errors++;
        if (!setFieldError(age.parentElement, age.value !== "")) errors++;
        if (!setFieldError(gender.parentElement, gender.value !== "")) errors++;
        if (!setFieldError(ethnicity.parentElement, ethnicity.value !== "")) errors++;
        if (!setFieldError(aids.parentElement, aids.value !== "")) errors++;

        // Update the banner with the live error count, or hide it when clean.
        if (errors > 0) {
            setupBanner.textContent =
                "There are " + errors + " error" + (errors === 1 ? "" : "s") +
                " on this screen. Please fix them to continue.";
            setupBanner.classList.remove("hidden");
        } else {
            setupBanner.classList.add("hidden");
        }

        return errors;
    }

    // Handle the Start Session press: validate, then begin if clean.
    function onStartSession() {

        // Mark that a submit was attempted, so fields now re-validate live.
        setupSubmitted = true;

        // Stop and show errors when any field is invalid.
        if (validateSetup() > 0) {
            return;
        }

        // All valid: store the setup data for this session's CSV row.
        sessionSetup = {
            participant_code: document.getElementById("su-code").value.trim(),
            condition: document.getElementById("su-condition").value,
            age_group: document.getElementById("su-age").value,
            gender: document.getElementById("su-gender").value,
            ethnicity: document.getElementById("su-ethnicity").value,
            visual_aids: document.getElementById("su-aids").value,
        };

        // Add the browser/device string for the session record.
        sessionSetup.device_browser = navigator.userAgent;

        // Send the setup to the server, which holds it for this session.
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: "session_setup", data: sessionSetup }));
        }

        // Hide the setup screen and reveal the normal camera flow.
        setupScreen.classList.add("hidden");
    }

    // Wire the Start Session button.
    startSessionBtn.addEventListener("click", onStartSession);

    // Live error-clearing: once the user has tried to submit, re-validate as
    // they fix each field, so errors clear and the count updates live. Only
    // active after the first submit attempt, so it does not flag empty fields
    // before the researcher has started filling the form.
    ["su-code", "su-condition", "su-age", "su-gender", "su-ethnicity", "su-aids"]
        .forEach((id) => {
            const el = document.getElementById(id);
            const event = el.tagName === "SELECT" ? "change" : "input";
            el.addEventListener(event, () => {
                if (setupSubmitted) validateSetup();
            });
        });


    // Run the record handler when the button is pressed.
    recordButton.addEventListener("click", onRecord);

    // Run the Start New handler when its button is pressed.
    document.getElementById("startnew-btn").addEventListener("click", onStartNew);

    // Start the camera and open the socket once the page has loaded.
    window.addEventListener("load", () => {
        startCamera();
        openSocket();
    });
    
    // Start New on the error panel: delete this recording and restart. 
    arStartNewBtn.addEventListener("click", onStartNew);

    // Start Prompts: hide the after-record panel and begin the setup check
    // for the prompts, reusing the entry-gate style.
    startPromptsBtn.addEventListener("click", onStartPrompts);

    // Run the record/finish handler on the prompt panel button.
    recordPromptsBtn.addEventListener("click", onRecordPrompts);

    // Begin the setup check before the prompts.
    function onStartPrompts() {

        // Hide the after-recording panel.
        afterRecord.classList.add("hidden");

        // Show the top "Prompts Check" legend.
        promptCheckLegend.classList.remove("hidden");

        // Show the middle pill for coaching prompts.
        message.classList.remove("hidden");

        // Show the bottom legend, same as the entry gate.
        legend.textContent = "Checking your position…";
        legend.classList.remove("hidden");

        // Make sure frames are being sent (they may have stopped).
        checking = true;

        // Tell the server to begin prompt-setup checking.
        socket.send("start_prompt_setup");

        // Show the face guide oval during the prompt setup and challenge.
        faceOval.classList.remove("hidden");
    }

    // Holds the challenge countdown timer so it can be stopped.
    let challengeCountdownTimer = null;

    // Replace the instruction with a 15 second countdown while recording.
    function startChallengeCountdown() {

        // The instruction element that becomes the timer.
        const box = document.getElementById("prompt-instruction");

        // Seconds remaining.
        let remaining = 15;

        // Show the first value with a small clock symbol.
        box.textContent = "⏱ " + remaining + "s";

        // Step down once per second.
        challengeCountdownTimer = setInterval(() => {

            // Take off one second.
            remaining -= 1;

            // Show the new value, never below zero.
            box.textContent = "⏱ " + Math.max(remaining, 0) + "s";

            // Stop at zero; the server also finishes at fifteen seconds.
            if (remaining <= 0) {
                clearInterval(challengeCountdownTimer);
                challengeCountdownTimer = null;
            }
        }, 1000);
    }

    // Stop the challenge countdown, if running.
    function stopChallengeCountdown() {

        // Clear the timer when it is active.
        if (challengeCountdownTimer) {
            clearInterval(challengeCountdownTimer);
            challengeCountdownTimer = null;
        }
    }

    // Run the Try Again handler when its button is pressed.
    document.getElementById("try-again-btn").addEventListener("click", onTryAgain);

    // Move from the Failed screen to the fresh set of prompts.
    function onTryAgain() {

        // Hide the Failed screen.
        document.getElementById("failed-screen").classList.add("hidden");

        // Clear the swap warning and attempts legend.
        swapWarning.classList.add("hidden");
        attemptsLegend.classList.add("hidden");

        // Show the new prompt panel with the prompts kept aside.
        if (pendingPrompts) {
            showPromptPanel(pendingPrompts);
            pendingPrompts = null;
        }
        
    }

})();
