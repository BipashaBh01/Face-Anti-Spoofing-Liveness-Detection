# ============================================================================
#  CHL_scoring.py — scores a recorded video with the trained FAS model
# ============================================================================
#  Loads the trained model once, then scores each finished recording and
#  returns a LIVE or SPOOF verdict.
#
#  This reuses the exact inference code from the training project:
#    build_model_inputs -> score_video -> check_video
#  so the scoring here matches the notebook results.
#
#  The heavy model is loaded a single time when the server starts, then
#  reused for every recording, so no time is lost reloading it each time.
# ============================================================================

from __future__ import annotations

# sys lets the trained project's folder be added to the import path.
import sys

# Path handles file and folder locations.
from pathlib import Path

# json reads the saved decision threshold.
import json

# torch loads the checkpoint and runs the model.
import torch

# Removes a DataParallel prefix from saved weight names if present.
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# Makes a clean copy of the weights before editing names.
from copy import deepcopy

from Functions.CHL_config import PROJECT_ROOT   # folder holding main_fas.py and Functions/


# ============================================================================
#  PATHS — REPLACE THESE THREE WITH THE REAL LOCATIONS ON THIS MACHINE
# ============================================================================

# 1. The trained project's root folder. It must contain lib_imports.py and
#    the Functions package (config, face_detect_2, FFT_3, optical_flow_4,
#    full_model_pipeline_13, challenge_response). Adding it to the path lets
#    the model code and its imports resolve.

# The trained project root is the app root: its Functions/ package holds
# config, full_model_pipeline_13 and challenge_response.
FAS_PROJECT_ROOT = PROJECT_ROOT

# The trained model checkpoint and its saved decision-threshold metrics.
CHECKPOINT_PATH   = PROJECT_ROOT / "output_combinedData" / "model_1" / "best_model_combined.pth"
TEST_METRICS_PATH = PROJECT_ROOT / "output_combinedData" / "model_1" / "held_out_test_metrics.json"


# ============================================================================
#  MAKE THE TRAINED PROJECT IMPORTABLE
# ============================================================================

# Add the project root to the front of the import path so its modules load.
sys.path.insert(0, str(FAS_PROJECT_ROOT))

# The device (cuda, mps or cpu) chosen by the trained project's config.
from Functions.config import DEVICE

# The model class.
from Functions.full_model_pipeline_13 import HybridPADModel

# The one-call scoring function: video in, verdict dict out.
from Functions.challenge_response import check_video

# threading runs the model load on its own thread and signals when ready.
import threading


# ============================================================================
#  DECISION SETTINGS
# ============================================================================

# The model's positive class (label 1) is SPOOF.
SPOOF_IS_POSITIVE = True


# ============================================================================
#  LOADING THE CHECKPOINT
# ============================================================================

def _looks_like_state_dict(value) -> bool:
    """Return True when the value looks like a model state_dict."""

    # A state_dict is a non-empty dict of string names to tensors.
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(isinstance(key, str) for key in value.keys())
        and all(torch.is_tensor(item) for item in value.values())
    )


def _extract_state_dict(checkpoint) -> dict:
    """Find the model weights inside the loaded checkpoint object."""

    # Case 1: the checkpoint is already a plain state_dict.
    if _looks_like_state_dict(checkpoint):
        return checkpoint

    # Case 2: the weights sit under a common key inside a larger dict.
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_weights"):
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                return candidate

    # Nothing usable was found.
    raise ValueError("No model state_dict found inside the checkpoint.")


def _load_threshold() -> float:
    """Read the decision threshold from the test-metrics JSON."""

    # Stop early with a clear message when the file is missing.
    if not TEST_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Test metrics not found:\n{TEST_METRICS_PATH}")

    # Read the JSON contents.
    with TEST_METRICS_PATH.open(mode="r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    # The threshold is stored under this key.
    if "best_threshold" not in metrics:
        raise KeyError("'best_threshold' missing from the test-metrics JSON.")

    # Return it as a float.
    return float(metrics["best_threshold"])


# ============================================================================
#  BUILD THE MODEL ONCE
# ============================================================================

# Holds the loaded model and threshold, filled in by the loader.
_model = None
_threshold = None
 
# Set once the model has finished loading, so scoring can wait for it.
_ready_event = threading.Event()
 
# Holds the error text if loading failed, so it can be reported.
_load_error: str | None = None


def load_model() -> None:
    """Load the model and threshold once, ready for scoring."""

    # Reach the module-level holders.
    global _model, _threshold

    # Do nothing if already loaded.
    if _model is not None:
        return

    # Stop early with a clear message when the checkpoint is missing.
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found:\n{CHECKPOINT_PATH}")

    # Load the checkpoint onto CPU first, keeping extra saved objects.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Pull the model weights out of whatever wrapper they came in.
    state_dict = _extract_state_dict(checkpoint)

    # Copy the weights, then drop any DataParallel "module." name prefix.
    clean_state = deepcopy(state_dict)
    consume_prefix_in_state_dict_if_present(clean_state, prefix="module.")

    # Build a fresh model on the chosen device.
    model = HybridPADModel().to(DEVICE)

    # Load the trained weights; names must match the model exactly.
    model.load_state_dict(clean_state, strict=True)

    # Move to the device again and switch to inference mode.
    model = model.to(DEVICE)
    model.eval()

    # Store the model and the decision threshold for reuse.
    _model = model
    _threshold = _load_threshold()

def start_background_load() -> None:
    """Begin loading the model on a background thread and return at once."""
 
    # Define the work the thread will do.
    def _worker():
 
        # Reach the module-level error holder.
        global _load_error
 
        try:
            # Load the model using the existing loader.
            load_model()
 
        except Exception as error:
            # Remember the exact error text for reporting later.
            _load_error = str(error)
 
            # Print the full error to the terminal for debugging.
            import traceback
            print("MODEL LOAD FAILED:")
            traceback.print_exc()
 
        finally:
            # Signal that loading has finished, success or not.
            _ready_event.set()
 
    # Start the worker on a daemon thread so it never blocks shutdown.
    threading.Thread(target=_worker, daemon=True).start()
 
 
def wait_until_ready(timeout: float | None = None) -> bool:
    """Wait for loading to finish. Return True when the model is ready."""
 
    # Block until the ready event is set, or the timeout passes.
    _ready_event.wait(timeout=timeout)
 
    # Ready only when loading finished with no error and a model exists.
    return _load_error is None and _model is not None
 
 
def get_load_error() -> str | None:
    """Return the load error text, or None when loading was fine."""
 
    # Simply hand back the stored error.
    return _load_error


# ============================================================================
#  SCORE ONE RECORDING
# ============================================================================

def score_recording(video_path) -> dict:
    """Score one recorded video and return the verdict dict.

    Returns the dict from the trained project's scoring, which includes:
        'verdict'    : 'SPOOF' or 'LIVE'
        'spoof_prob' : probability the video is a spoof
        'live_prob'  : probability the video is live
        'usable_frac': fraction of frames that were usable
    """

    # Make sure the model is loaded before scoring.
    load_model()

    # Run the trained one-call scoring function on the video.
    result = check_video(
        str(video_path),
        threshold=_threshold,
        spoof_is_positive=SPOOF_IS_POSITIVE,
        model=_model,
        device=DEVICE,
        verbose_prep=True,
        verbose_score=True,
    )

    # Hand back the verdict dict.
    return result