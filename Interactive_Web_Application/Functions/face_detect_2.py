# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
# from Functions.config_SiW import *
from Functions.config import *

#========================================================================================================================================================
"""
Create a face detector object.  
This object will be used to detect and crop faces from images (video frames).
"""

# ------------------------------------------------------------
# Choose ONNXRuntime execution provider
# ------------------------------------------------------------
# SCRFD does not use torch.device directly. It uses ONNXRuntime providers instead.
# If CUDAExecutionProvider is available, SCRFD can run on GPU. Otherwise, it will safely fall back to CPU.
# ------------------------------------------------------------
ort_version = ort.__version__
available_providers = ort.get_available_providers()
print("ORT Version:", ort_version)
print("Available ONNXRuntime providers:", available_providers)

if "CUDAExecutionProvider" in available_providers:
    # SCRFD_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    SCRFD_PROVIDERS = ["CUDAExecutionProvider"]
    SCRFD_CTX_ID = 0       # 0 means use GPU device 0
else:
    SCRFD_PROVIDERS = ["CPUExecutionProvider"]
    SCRFD_CTX_ID = -1      # -1 means CPU


#=============================================================================
# MAIN FUNCTION TO CALL THE FACE DETECTOR
#=============================================================================

def init_face_detector(providers=SCRFD_PROVIDERS,
                       ctx_id=SCRFD_CTX_ID,
                       root="/storage/research/csdata_L/xp814142/insightface_models"):
    """
    Create and prepare the SCRFD face detector (InsightFace).
    Returns a ready-to-use detector object.

    Create this ONCE, then pass the returned object to any function
    that needs face detection.
    """
    
    # ------------------------------------------------------------
    # Create SCRFD detector through InsightFace
    # ------------------------------------------------------------
    # name="buffalo_l": Loads InsightFace model pack.
    # allowed_modules=["detection"]: Loads only the face detection module. This avoids loading recognition / age / gender models unnecessarily.
    # providers: Tells ONNXRuntime whether to use GPU or CPU.
    # ------------------------------------------------------------
    detector = FaceAnalysis(
        name="buffalo_l",
        root=root,
        allowed_modules=["detection"],
        providers=providers
    )

    # ------------------------------------------------------------
    # Prepare detector
    # ------------------------------------------------------------
    # ctx_id: 0  -> GPU | -1 -> CPU
    # det_size: Input size used by SCRFD detector. This controls detection quality/speed, not the final crop size.
    # det_thresh: Minimum detection confidence. This is similar in purpose to MTCNN thresholds, but SCRFD has one main detection threshold instead of three stage thresholds.
    # ------------------------------------------------------------
    detector.prepare(
        ctx_id=ctx_id,
        det_size=CFG.SCRFD_DET_SIZE,
        det_thresh=CFG.SCRFD_DET_THRESH
    )

    return detector

# Call the face detector
face_detector = init_face_detector()

#========================================================================================================================================================


#========================================================================================================================================================

#=============================================================================
# MAIN FRAME SAMPLING FUNCTION
#=============================================================================

def sample_frames_from_video(video_path, n_frames=CFG.N_FRAMES, verbose=False):
    """
    Sample evenly spaced frames from a video using Decord.
    It:
        - Opens a video file.
        - Gets the total number of frames.
        - Selects evenly spaced frame indices across the full video.
        - Reads those selected frames.
        - Stores them in a Python list.
        - Pads missing frames if fewer than n_frames were successfully read.
        - Returns exactly n_frames frames.

    Parameters
    ----------
    video_path : str or pathlib.Path
        Path to the input video file.
    n_frames : int, default=CFG.N_FRAMES
        Number of frames to sample from the video.
    verbose : bool, default=False
        If True, print device/decoding/fallback messages (useful for debugging
        a single video). If False (default), run SILENTLY so batch loops like
        precompute keep a clean single progress bar.

    Returns
    -------
    frames : list of np.ndarray
        List containing exactly n_frames RGB frames.
        Each frame has shape (height, width, 3), dtype uint8.

    Notes
    -----
    - Decord returns RGB frames, so BGR-to-RGB conversion is not needed.
    - Tries the device in DEVICE, but only attempts GPU if this Decord build
      actually supports it (otherwise it goes straight to CPU, no failed retry).
    - If GPU decoding fails, it falls back to CPU.
    - If fewer than n_frames are read, the last valid frame is repeated; if none
      are read, black frames are added.
    - Reports corrupted/unreadable files (only when verbose=True).
    """

    # Use existing DEVICE variable
    preferred_device = str(DEVICE)

    if verbose:
        print(f"Sampler ready. Requested device: {preferred_device}")

    def _read_with_device(device_to_use):
        """Internal helper: read frames using the selected device."""
        # Choose the Decord context based on device
        if "cuda" in device_to_use.lower() or "gpu" in device_to_use.lower():
            ctx = gpu(0)
            active_device = "cuda (gpu:0)"
        else:
            ctx = cpu(0)
            active_device = "cpu"

        if verbose:
            print(f"[Sampler] Actually decoding on device: {active_device}")

        # Open video file using Decord
        # If the file is corrupted/unreadable, this line typically raises exception
        vr = VideoReader(str(video_path), ctx=ctx)

        # Get total number of frames in the video
        total = int(len(vr))

        # If no frames exist (readable-but-empty/corrupted), return empty list
        if total < 1:
            if verbose:
                print(f"[Sampler] WARNING: No frames found (possibly corrupted): {video_path}")
            return []

        # Select evenly spaced frame positions from the full video
        indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()

        frames = []

        # Read selected frames in one batch; get_batch returns a Decord NDArray
        frame_batch = vr.get_batch(indices)

        # Convert to NumPy (shape: T, H, W, C); .asnumpy() handles CPU & GPU
        frames_np = frame_batch.asnumpy()

        # Store each frame (Decord already gives RGB)
        for frame in frames_np:
            frames.append(frame.astype(np.uint8))

        return frames

    try:
        # First try using requested DEVICE
        frames = _read_with_device(preferred_device)

    except Exception as e:
        if verbose:
            print(f"[Sampler] Decord failed on device '{preferred_device}' for: {video_path}")
            print(f"[Sampler] Error: {e}")

        # If GPU fails, retry on CPU
        if "cuda" in preferred_device.lower() or "gpu" in preferred_device.lower():
            if verbose:
                print("[Sampler] Retrying on CPU...")

            try:
                frames = _read_with_device("cpu")

            except Exception as cpu_error:
                # Failed on BOTH GPU and CPU -> treat as corrupted/unreadable
                if verbose:
                    print(f"[Sampler] CORRUPTED OR UNREADABLE FILE (failed on GPU and CPU): {video_path}")
                    print(f"[Sampler] CPU error: {cpu_error}")
                return []
        else:
            # Failed on CPU directly (no GPU to fall back to) -> corrupted/unreadable
            if verbose:
                print(f"[Sampler] CORRUPTED OR UNREADABLE FILE: {video_path}")
            return []

    # If fewer frames than needed were successfully read, pad to n_frames
    while len(frames) < n_frames:
        if len(frames) > 0:
            # Repeat the last valid frame
            frames.append(frames[-1])
        else:
            # If no frame was read at all, add a black image
            frames.append(
                np.zeros((CFG.FRAME_SIZE, CFG.FRAME_SIZE, 3), dtype=np.uint8)
            )

    # Return exactly n_frames
    return frames[:n_frames]


#=============================================================================
## Demo Tests for the farme sampling function
#=============================================================================
# Approach 1 - Real end-to-end test using an OpenCV-generated video
# OpenCV ships its own video codecs. 
# We create a throwaway video, then run the real sampler on it to confirm the whole pipeline works.

def make_test_video(path, n=30, size=128):
    """
    Write a small synthetic video to `path`.

    Each frame is a solid gray that gets brighter over time, so frames are
    visually distinct — that lets us verify the sampler is actually reading
    different positions (not the same frame repeatedly).
    """
    # "mp4v" is a codec bundled with OpenCV.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Arguments: path, codec, frames-per-second, (width, height).
    vw = cv2.VideoWriter(path, fourcc, 10.0, (size, size))

    for i in range(n):
        # Build one frame: a solid color whose brightness scales with `i`.
        # Frame 0 is black-ish, the last frame is near-white.
        shade = i * (255 // n)
        frame = np.full((size, size, 3), shade, dtype=np.uint8)
        vw.write(frame)  # append this frame to the video

    vw.release()  # finalise and close the file (flushes to disk)


def test_end_to_end():
    """Full pipeline: write a real video, decode it, check the output shape."""
    # Put the test file in the OS temp directory so it gets cleaned up easily.
    # In this step: a string path is built, something like /tmp/test.mp4. 
    # It's pure text. Nothing is written to disk, no video exists yet.
    tmp = os.path.join(tempfile.gettempdir(), "test.mp4")

    # The actual video file (with 30 frames) only gets created on this step
    make_test_video(tmp)

    # Ask the sampler for 8 frames out of the 30 we wrote.
    frames = sample_frames_from_video(tmp, n_frames=8)

    # The three guarantees the function promises:
    assert len(frames) == 8                      # exactly n_frames returned
    assert frames[0].shape == (128, 128, 3)      # correct (H, W, C) shape
    assert frames[0].dtype == np.uint8           # correct pixel dtype

    print("Approach 1 (end-to-end) passed.")



# Approach 2 - Pure unit test by mocking Decord's VideoReader
# Here we never touch a real file or codec. We replace VideoReader with a fake
# object that returns frames we control. This isolates and tests the sampler's
# OWN logic (sampling + padding) without any video I/O — fast and reliable.

def test_padding_when_short():
    """
    Simulate a video with only 3 frames while asking for 8.

    Expectation: the sampler reads the 3 real frames, then pads up to 8 by
    repeating the last valid frame.
    """
    # --- Build the fake VideoReader ---------------------------------------
    fake = MagicMock()

    # len(vr) inside the function should report 3 frames available.
    fake.__len__.return_value = 3

    # get_batch(...) returns an object with an .asnumpy() method. We make that
    # return 3 distinct frames (filled with 10, 20, 30) of shape (3, 64, 64, 3).
    batch = MagicMock()
    batch.asnumpy.return_value = np.stack([
        np.full((64, 64, 3), k, np.uint8) for k in (10, 20, 30)
    ])
    fake.get_batch.return_value = batch

    # --- Patch the objects that sample_frames_from_video() uses internally  ---------
    # VideoReader is replaced with a fake video reader, so the test does not need
    # a real video file. cpu and gpu are patched to avoid real hardware/context calls.
    with patch("Functions.face_detect_2.VideoReader", return_value=fake), \
         patch("Functions.face_detect_2.cpu"), \
         patch("Functions.face_detect_2.gpu"):
        out = sample_frames_from_video("dummy.mp4", n_frames=8)

    # --- Verify the padding behavior --------------------------------------
    assert len(out) == 8                       # padded up to the requested count
    assert np.array_equal(out[-1], out[2])     # last slot == last real frame (repeated)

    print("Approach 2 (padding) passed.")


# Approach 3 - Edge cases: empty video and corrupted file
def test_empty_video():
    """
    A readable file that reports 0 frames should yield an empty list.

    (The outer function will normally pad, but _read_with_device returns []
    early on total < 1; this test targets that early-exit path directly via
    a fake that has no frames AND a get_batch that would fail if called.)
    """
    fake = MagicMock()
    fake.__len__.return_value = 0  # zero frames -> triggers the early warning/return

    with patch("Functions.face_detect_2.VideoReader", return_value=fake), \
         patch("Functions.face_detect_2.cpu"), \
         patch("Functions.face_detect_2.gpu"):
        out = sample_frames_from_video("empty.mp4", n_frames=8)

    # With zero real frames, padding fills all 8 slots with black frames.
    assert len(out) == 8
    assert np.array_equal(out[0], np.zeros_like(out[0]))  # black placeholder

    print("Approach 3a (empty video) passed.")


def test_corrupted_file():
    """
    A file that can't be opened should make VideoReader raise.

    We simulate the CPU-only case (no GPU fallback), where a raise means the
    function reports corruption and returns an empty list.
    """
    # Exception makes any call to VideoReader(...) raise.
    with patch("Functions.face_detect_2.VideoReader", side_effect=Exception("bad file")), \
         patch("Functions.face_detect_2.cpu"), patch("__main__.gpu"), \
         patch("Functions.face_detect_2.DEVICE", "cpu"):  # force CPU path so there's no retry
        out = sample_frames_from_video("broken.mp4", n_frames=8)

    # On total failure the function returns [] (no padding happens).
    assert out == []

    print("Approach 3b (corrupted file) passed.")

#========================================================================================================================================================


#========================================================================================================================================================

#=============================================================================
# MAIN FRAME EXTRACTION FUNCTION
#=============================================================================

def extract_face_or_resize(frame, size=CFG.FRAME_SIZE, margin=0.2):
    """
    Per-frame face DETECTION + landmark capture for the anti-spoofing pipeline.

    This function does NOT make the final align-vs-letterbox decision — that is
    made per video by the sequence-level stage, which needs to see all frames
    (to interpolate short landmark gaps from neighbours and keep one consistent
    geometry per video). Here we only:
      - detect the largest face,
      - capture its RAW full-frame landmarks (alignment needs these),
      - run a sanity check and flag whether the landmarks are trustworthy,
      - record everything in meta.

    A frame is NEVER dropped; meta always describes what was found.

    Parameters
    ----------
    frame : np.ndarray
        Input video frame in RGB format, shape (H, W, 3), uint8.
    size : int, default=CFG.FRAME_SIZE
        Target square size for the later resize stage (kept for reference).
    margin : float, default=0.2
        Margin used later by the letterbox-fallback crop (not applied here).

    Returns
    -------
    meta : dict
        Diagnostics for this frame:
            face_found  : bool   -> a face was detected
            n_faces     : int    -> number of faces SCRFD returned
            score       : float  -> detection confidence of the chosen face
            bbox        : tuple | None -> clamped (x1, y1, x2, y2) of the face
            kps         : np.ndarray | None -> RAW (5, 2) landmarks in FULL-FRAME
                          coordinates (None if no face)
            kps_ok      : bool   -> landmarks present AND passed sanity check
            reason      : str    -> 'face' | 'no_face' | 'error'

    Notes
    -----
    - Returns ONLY meta (no image): the actual 224x224 face is produced later by
      the per-video stage, which decides align-or-letterbox for the whole clip.
    - SCRFD expects BGR; we convert for detection only.
    - kps are RAW full-frame coords so InsightFace norm_crop can align directly
      from the raw frame in a single warp (no double resize).
    """
    # Default metadata; overwritten as we learn about the frame.
    meta = {"face_found": False, "n_faces": 0, "score": 0.0, "bbox": None,
            "kps": None, "kps_ok": False, "detection": "no_face_found", "kps_reason": None}

    h, w = frame.shape[:2]

    # SCRFD/InsightFace expects BGR, but our frame is RGB -> convert for detection.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        # Run SCRFD detection on the GPU (returns a list of Face objects).
        faces = face_detector.get(bgr)
        meta["n_faces"] = len(faces)

        if faces:
            # Largest face by box area (most likely the subject).
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            meta["score"] = float(getattr(best, "det_score", 0.0))

            # Clamp the box to image bounds; store it for the fallback crop.
            x1, y1, x2, y2 = best.bbox
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(w, int(x2)), min(h, int(y2))
            meta["bbox"] = (cx1, cy1, cx2, cy2)

            # Capture RAW full-frame landmarks (what alignment will use later).
            kps = getattr(best, "kps", None)
            if kps is not None:
                kps = np.asarray(kps, dtype=np.float32)
            meta["kps"] = kps

            # Flag whether these landmarks are trustworthy for alignment.
            # meta["kps_ok"] = _landmarks_ok(kps, cx1, cy1, cx2, cy2)
            bool_result, reason = valid_5pt_landmarks(kps, meta["bbox"], debug=True)
            meta["kps_ok"] = bool_result
            meta["kps_reason"] = reason   # store why it failed/passed
            meta["face_found"] = True
            meta["detection"] = "face_found"

    except Exception as e:
        meta["reason"] = "error"
        print(f"[face] detection failed: {e}")

    return meta



#----------------------------------------------------------------------------------------
# LETTERBOX
#----------------------------------------------------------------------------------------

# Shared helper — used by BOTH extract_face_or_resize and compute_freq_map
def letterbox(img, target, pad_value=128, return_transform=False):
    """
    Resize img to target x target WITHOUT distortion, padding leftover space
    with a flat color. Handles both 3-channel (RGB) and 1-channel (grayscale)
    input so every stage of the pipeline resizes identically.

    Parameters
    ----------
    img : np.ndarray
        Image of shape (H, W, C) or (H, W).
    target : int
        Final square output size in pixels.
    pad_value : int, default=128
        Gray fill value for padding. Keep identical across all stages and
        match whatever the model was trained with.
    return_transform : bool, default=False
        If True, also return (scale, left, top): the factor the image was
        scaled by and the centering offsets. A point (x, y) in the input maps
        to (x*scale + left, y*scale + top) in the output — used to move
        landmarks into the letterboxed coordinate system.

    Returns
    -------
    np.ndarray, shape (target, target, C), uint8.
    (If return_transform=True: (canvas, (scale, left, top)).)
    """
    if img.ndim == 2:                                   # grayscale -> add channel axis
        img = img[:, :, None]
    ih, iw = img.shape[:2]                              # current height/width
    c = img.shape[2]                                    # channel count (1 or 3)
    if ih == 0 or iw == 0:                              # safety: empty input
        canvas = np.full((target, target, c), pad_value, np.uint8)
        return (canvas, (1.0, 0, 0)) if return_transform else canvas
    scale = min(target / iw, target / ih)               # largest scale that fits
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh))                 # aspect-preserving resize
    if resized.ndim == 2:                               # cv2 may drop channel axis
        resized = resized[:, :, None]
    canvas = np.full((target, target, c), pad_value, np.uint8)  # padded background
    top, left = (target - nh) // 2, (target - nw) // 2  # center offsets
    canvas[top:top + nh, left:left + nw] = resized      # paste centered
    if return_transform:                                # also hand back the transform
        return canvas, (scale, left, top)
    return canvas


#----------------------------------------------------------------------------------------
# ALIGNMENT
#----------------------------------------------------------------------------------------

def valid_5pt_landmarks(
    kps,
    bbox,
    box_pad_ratio=CFG.LMK_BOX_PAD_RATIO,
    min_eye_dx_ratio=CFG.LMK_MIN_EYE_DX_RATIO,
    max_eye_dx_ratio=CFG.LMK_MAX_EYE_DX_RATIO,
    min_mouth_dx_ratio=CFG.LMK_MIN_MOUTH_DX_RATIO,
    max_eye_roll_deg=CFG.LMK_MAX_EYE_ROLL_DEG,
    max_mouth_roll_deg=CFG.LMK_MAX_MOUTH_ROLL_DEG,
    min_eye_to_mouth_ratio=CFG.LMK_MIN_EYE_TO_MOUTH_RATIO,
    max_eye_to_mouth_ratio=CFG.LMK_MAX_EYE_TO_MOUTH_RATIO,
    max_nose_offset_ratio=CFG.LMK_MAX_NOSE_OFFSET_RATIO,
    debug=False,
):
    """
    Check whether 5 face landmarks are reliable enough for face alignment.

    Expected landmark order:
        0 = left eye
        1 = right eye
        2 = nose
        3 = left mouth corner
        4 = right mouth corner

    bbox:
        (x1, y1, x2, y2)

    Parameters
    ----------
    debug : bool, default=False
        If False, returns just a bool (valid or not).
        If True, returns (bool, reason) where reason is a short string naming
        the check that failed ('ok' if it passed). Useful for profiling which
        check trips most often so thresholds can be tuned from data.

    Returns
    -------
    bool                      (when debug=False)
    (bool, reason_str)        (when debug=True)
    """

    # Small helper so every exit point respects the debug flag.
    def _result(ok, reason):
        return (ok, reason) if debug else ok

    # ------------------------------------------------------------
    # 1. Landmarks must exist and must contain exactly 5 points
    # ------------------------------------------------------------
    if kps is None:
        return _result(False, "kps_none")

    kps = np.asarray(kps, dtype=np.float32)

    if kps.shape != (5, 2):
        return _result(False, "bad_shape")

    # ------------------------------------------------------------
    # 2. Landmark values must be real numbers
    # ------------------------------------------------------------
    if not np.all(np.isfinite(kps)):
        return _result(False, "non_finite")

    # ------------------------------------------------------------
    # 3. Face box must be valid
    # ------------------------------------------------------------
    x1, y1, x2, y2 = map(float, bbox)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 1 or box_h <= 1:
        return _result(False, "bad_box")

    # ------------------------------------------------------------
    # 4. Unpack landmarks
    # ------------------------------------------------------------
    le, re, nose, lm, rm = kps

    # le = left eye
    # re = right eye
    # nose = nose tip
    # lm = left mouth corner
    # rm = right mouth corner

    # ------------------------------------------------------------
    # 5. Landmarks should be inside or near the face box
    # ------------------------------------------------------------
    pad_x = box_pad_ratio * box_w
    pad_y = box_pad_ratio * box_h

    for px, py in kps:
        inside_x = (x1 - pad_x) <= px <= (x2 + pad_x)
        inside_y = (y1 - pad_y) <= py <= (y2 + pad_y)

        if not (inside_x and inside_y):
            return _result(False, "outside_box")

    # ------------------------------------------------------------
    # 6. Left/right order should make sense
    # ------------------------------------------------------------
    if le[0] >= re[0]:
        return _result(False, "eye_order")

    if lm[0] >= rm[0]:
        return _result(False, "mouth_order")

    # ------------------------------------------------------------
    # 7. Eye distance should be reasonable
    # ------------------------------------------------------------
    eye_dx = re[0] - le[0]
    eye_dy = abs(re[1] - le[1])

    if eye_dx < min_eye_dx_ratio * box_w:
        return _result(False, "eye_dx_small")

    if eye_dx > max_eye_dx_ratio * box_w:
        return _result(False, "eye_dx_large")

    # ------------------------------------------------------------
    # 8. Eye line should not be too tilted
    # ------------------------------------------------------------
    eye_roll_deg = np.degrees(np.arctan2(eye_dy, eye_dx))

    if eye_roll_deg > max_eye_roll_deg:
        return _result(False, "eye_roll")

    # ------------------------------------------------------------
    # 9. Mouth width should be reasonable
    # ------------------------------------------------------------
    mouth_dx = rm[0] - lm[0]
    mouth_dy = abs(rm[1] - lm[1])

    if mouth_dx < min_mouth_dx_ratio * box_w:
        return _result(False, "mouth_dx_small")

    # ------------------------------------------------------------
    # 10. Mouth line should not be too tilted
    # ------------------------------------------------------------
    mouth_roll_deg = np.degrees(np.arctan2(mouth_dy, mouth_dx))

    if mouth_roll_deg > max_mouth_roll_deg:
        return _result(False, "mouth_roll")

    # ------------------------------------------------------------
    # 11. Nose should be vertically between eyes and mouth
    # ------------------------------------------------------------
    eye_y = (le[1] + re[1]) / 2.0
    mouth_y = (lm[1] + rm[1]) / 2.0
    nose_y = nose[1]

    if not (eye_y <= nose_y <= mouth_y):
        return _result(False, "nose_vertical")

    # ------------------------------------------------------------
    # 12. Eye-to-mouth vertical distance should be reasonable
    # ------------------------------------------------------------
    eye_to_mouth_y = mouth_y - eye_y

    if eye_to_mouth_y < min_eye_to_mouth_ratio * box_h:
        return _result(False, "eye_mouth_small")

    if eye_to_mouth_y > max_eye_to_mouth_ratio * box_h:
        return _result(False, "eye_mouth_large")

    # ------------------------------------------------------------
    # 13. Nose should not be horizontally too far from eye centre
    # ------------------------------------------------------------
    eye_center_x = (le[0] + re[0]) / 2.0
    nose_offset = abs(nose[0] - eye_center_x)

    if nose_offset > max_nose_offset_ratio * eye_dx:
        return _result(False, "nose_offset")

    # ------------------------------------------------------------
    # If all checks passed, landmarks are safe enough for alignment
    # ------------------------------------------------------------
    return _result(True, "all_landmarks_ok")

#========================================================================================================================================================

#========================================================================================================================================================

#=============================================================================
# MAIN FUNCTION TO PREPARE EACH FACE FROM VIDEO FRAMES
#=============================================================================

def _expand_box(bbox, margin, hw):
    """Expand a clamped (x1,y1,x2,y2) box by `margin` and re-clamp to image."""
    h, w = hw
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin)); y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin)); y2 = min(h, int(y2 + bh * margin))
    return x1, y1, x2, y2


def _canonical_kps(size):
    """The 5 ArcFace template points scaled to `size` (where aligned features land)."""
    base = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    return base * (size / 112.0)



def prepare_video_faces(
    frames,
    metas,
    size=CFG.FRAME_SIZE,
    margin=0.2,
    align_min_frac=CFG.ALIGN_MIN_FRAC,
    max_interp_gap=CFG.MAX_INTERP_GAP
):
    """
    Turn a whole video's frames into a CONSISTENT set of 224x224 faces.

    The align-vs-letterbox choice is made ONCE for the whole video (never
    mixed), so the RGB / FFT / optical-flow streams all share the same geometry
    and the gated fusion stays consistent. Steps:

      1. From per-frame metas, mark which frames have trustworthy landmarks.
      2. Interpolate SHORT gaps of bad frames (<= max_interp_gap) from their
         good neighbours; hold nearest-good landmarks at the video edges.
      3. If enough frames are usable after interpolation (>= align_min_frac),
         ALIGN every frame with InsightFace norm_crop. Otherwise LETTERBOX every
         frame. One decision, applied to all frames.
      4. Record the full decision trail in video_meta.

    A video is NEVER dropped — a broken video is letterboxed, not discarded, so
    spoofs that defeat detection are still seen by the model.

    Parameters
    ----------
    frames : list of np.ndarray
        The video's RGB frames, each (H, W, 3) uint8, in temporal order.
    metas : list of dict
        Per-frame meta from extract_face_or_resize (same length/order as frames).
    size : int, default=CFG.FRAME_SIZE
        Output square size (224).
    margin : float, default=0.2
        Margin for the letterbox-fallback crop around the face box.
    align_min_frac : float, default=CFG.ALIGN_MIN_FRAC or 0.7
        Minimum fraction of usable-after-interpolation frames to align the video.
    max_interp_gap : int, default=CFG.MAX_INTERP_GAP or 3
        Longest run of consecutive bad frames that may be interpolated.

    Returns
    -------
    faces : list of np.ndarray
        One (size, size, 3) uint8 face per input frame (aligned OR letterboxed,
        the same choice for the whole video).
    kps_list : list of np.ndarray or None
        Per-frame landmarks mapped into the output image's coordinates (for the
        liveness / flow-alignment stages). None where unavailable.
    video_meta : dict
        mode            : 'aligned' | 'letterbox'  -> the whole-video decision
        n_frames        : int
        n_good          : int    -> frames with trustworthy raw landmarks
        n_interpolated  : int    -> bad frames filled from neighbours
        n_unfilled      : int    -> bad frames that could not be filled
        usable_frac     : float  -> (good + interpolated) / n_frames
        bad_indices     : list   -> frames whose raw landmarks failed the check
        interp_indices  : list   -> frames whose landmarks were interpolated
    """
    n = len(frames)   # stores the number of frames in the video into n, e.g., 16.

    # --- 1. Collect raw landmarks and the good/bad flag per frame ---------
    kps_raw = [m.get("kps") if m.get("kps_ok") else None for m in metas]
    good = [k is not None for k in kps_raw]               # True where landmarks usable
    bad_indices = [i for i, g in enumerate(good) if not g]

    # --- 2. Interpolate short gaps of bad frames from good neighbours -----
    interp_indices = []
    i = 0
    while i < n:
        if good[i]:
            i += 1
            continue
        # Found the start of a bad run; find where it ends.
        start = i
        while i < n and not good[i]:
            i += 1
        end = i - 1                                       # inclusive end of bad run
        gap_len = end - start + 1

        prev_good = start - 1                             # nearest good before the run
        next_good = i if i < n else None                 # nearest good after the run

        # Only fill short gaps; longer runs stay unfilled (push toward letterbox).
        if gap_len <= max_interp_gap:     # maxmimum gap
            if prev_good >= 0 and next_good is not None:
                # Both sides good -> linearly interpolate between them.
                a, b = kps_raw[prev_good], kps_raw[next_good]
                span = next_good - prev_good
                for j in range(start, end + 1):
                    t = (j - prev_good) / span            # 0..1 position in the gap
                    kps_raw[j] = (1 - t) * a + t * b      # blended landmarks
                    good[j] = True
                    interp_indices.append(j)
            elif prev_good >= 0:
                # Gap at the END -> hold the last good landmarks.
                for j in range(start, end + 1):
                    kps_raw[j] = kps_raw[prev_good].copy()
                    good[j] = True
                    interp_indices.append(j)
            elif next_good is not None:
                # Gap at the START -> hold the first good landmarks.
                for j in range(start, end + 1):
                    kps_raw[j] = kps_raw[next_good].copy()
                    good[j] = True
                    interp_indices.append(j)
            # else: no good frame anywhere -> leave unfilled.

    n_good = sum(1 for m in metas if m.get("kps_ok"))
    n_interp = len(interp_indices)
    n_usable = sum(good)
    usable_frac = (n_usable / n) if n else 0.0

    # --- 3. One decision for the whole video ------------------------------
    mode = "aligned" if usable_frac >= align_min_frac else "letterbox"

    faces, kps_list = [], []
    for idx, frame in enumerate(frames):
        if mode == "aligned" and kps_raw[idx] is not None:
            # Align from the RAW frame in one warp using (real/interp) landmarks.
            face = face_align.norm_crop(frame, landmark=kps_raw[idx], image_size=size)
            faces.append(face)
            # In an aligned face, landmarks sit at the canonical template spots.
            kps_list.append(_canonical_kps(size))
        else:
            # Letterbox path: crop around the face box if we have one, else full frame.
            bbox = metas[idx].get("bbox")
            if bbox is not None:
                bx1, by1, bx2, by2 = _expand_box(bbox, margin, frame.shape[:2])
                crop = frame[by1:by2, bx1:bx2]
            else:
                crop = frame
            if crop.size == 0:
                crop = frame
            out, (scale, pad_left, pad_top) = letterbox(crop, size, return_transform=True)
            faces.append(out)
            # Map this frame's landmarks (if any) into the letterboxed image.
            k = metas[idx].get("kps")
            if k is not None and bbox is not None:
                k = np.asarray(k, dtype=np.float32).copy()
                k[:, 0] = (k[:, 0] - bx1) * scale + pad_left
                k[:, 1] = (k[:, 1] - by1) * scale + pad_top
                kps_list.append(k)
            else:
                kps_list.append(None)

    # --- 4. Decision trail ------------------------------------------------
    video_meta = {
        "mode": mode,
        "n_frames": n,
        "n_good": n_good,
        "n_interpolated": n_interp,
        "n_unfilled": n - sum(good),
        "usable_frac": usable_frac,
        "bad_indices": bad_indices,
        "interp_indices": interp_indices,
    }
    return faces, kps_list, video_meta


#=============================================================================
## Demo Tests for the above function
#=============================================================================

def _canonical_kps(size):
    base = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    return base * (size / 112.0)



# Test 1 -- visual: align one frame's face and check the canonical dots land right

def test_alignment_visual_from_video(video_path, size=CFG.FRAME_SIZE,
                                     out_path="./test_siW_files/aligned_check.png"):
    # Pull 8 RGB frames
    frames = sample_frames_from_video(video_path, n_frames=8)
    assert len(frames) > 0, "No frames returned from the video."     # Stop if none

    # Try frames until one has a detectable face (skips blank early frames) --> Take the first detectable frame only
    # Then convert it to BGR, run detector; save it as rgb and break.
    rgb = None
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)   # detector wants BGR
        if face_detector.get(bgr):
            rgb = f
            break
    assert rgb is not None, "No face detected in any sampled frame."

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # From the chosen frame's faces, pick the largest one
    best = max(face_detector.get(bgr),
               key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
    # Grab its 5 landmarks
    kps = np.asarray(best.kps, dtype=np.float32)

    # Align the rgb face to 224x224 using its landmarks
    aligned = face_align.norm_crop(rgb, landmark=kps, image_size=size)  # RGB

    # Draw the CANONICAL points; they should sit on eyes / nose / mouth.
    canon = _canonical_kps(size)

    # Convert aligned face to BGR for saving
    vis = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR).copy()

    # Draw a red dot at each canonical point
    for (x, y) in canon:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)  # red dots
    # Save the dotted image
    cv2.imwrite(out_path, vis)

    # Print saved location
    print(f"Test 1: saved '{out_path}'. Check the 5 red dots land on left eye, "
          f"right eye, nose, and the two mouth corners.")

    # Sanity-check shape and dtype
    assert aligned.shape == (size, size, 3) and aligned.dtype == np.uint8



# Test 2 -- functional: run the per-video pipeline on the sampled frames

def test_prepare_video_runs_from_video(video_path, size=CFG.FRAME_SIZE):
    # Pull 8 frames
    frames = sample_frames_from_video(video_path, n_frames=8)
    # Run per-frame detection on each
    metas = [extract_face_or_resize(f) for f in frames]
    # Run the per-video pipeline
    faces, kps_list, video_meta = prepare_video_faces(frames, metas, size=size)
    # Counts must match on the number of faces and kps_list
    assert len(faces) == len(frames) == len(kps_list)
    # Check each face is (224,224,3) uint8
    for f in faces:
        assert f.shape == (size, size, 3) and f.dtype == np.uint8
    # mode must be valid
    assert video_meta["mode"] in ("aligned", "letterbox")
    # Report the decision (mode, good count, interpolated count, usable fraction)
    print(f"Test 2: prepare_video_faces ran. mode={video_meta['mode']}, "
          f"good={video_meta['n_good']}, interp={video_meta['n_interpolated']}, "
          f"usable_frac={video_meta['usable_frac']:.2f}")

#========================================================================================================================================================