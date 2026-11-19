# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
# from Functions.config_1 import *
from Functions.face_detect_2 import *
from Functions.FFT_3 import *
from Functions.optical_flow_4 import *


"""
Before training begins, this function pre-computes all heavy CPU operations **once** and saves the results to disk.

During training, `__getitem__()` simply loads the cached `.npy` files instead of recomputing them — keeping the GPU fed with data and utilisation high.

| Stream | Operation | Saved As |
|--------|-----------|----------|
| RGB | Face extraction + crop | `frame_XXX.npy` |
| Frequency | FFT texture maps | `freq_XXX.npy` |
| Flow | Optical flow maps | `flow_XXX.npy` |


> If `CFG.N_FRAMES` or `CFG.FRAME_SIZE` is changed, delete the cache folder and re-run.
"""


#=============================================================================
# MAIN FUNCTION FOR PRECOMPUTE STEP
#=============================================================================

# =============================================================================
# STREAM PRE-COMPUTATION & CACHING
# =============================================================================
# WHY THIS EXISTS:
#   compute_freq_map() and compute_optical_flow_sequence() are heavy ops.
#   Running them inside __getitem__() means they execute for every sample on
#   every epoch, keeping the GPU idle and making training extremely slow.
#
#   This function pre-computes all freq maps and optical flow maps ONCE before
#   training begins, saving results as .npy files to disk. During training,
#   __getitem__() simply loads these cached arrays — making data loading fast
#   and keeping GPU utilisation high.
#
# OUTPUT CACHE STRUCTURE:
#   CFG.CACHE_DIR/
#   └── <video_id>/
#       ├── frame_000.npy       # Prepared (aligned/letterboxed) RGB face
#       ├── frame_001.npy
#       ├── freq_000.npy        # FFT frequency map for frame 0
#       ├── freq_001.npy
#       ├── flow_000.npy        # Optical flow map for frame 0
#       ├── flow_001.npy
#       ├── ...
#       └── video_meta.json     # Per-video decision trail (mode, interp, etc.)
# =============================================================================

def precompute_and_cache(dataframe, directory):
    """
    Pre-computes and caches prepared RGB faces, frequency maps, and optical
    flow maps for all videos in the dataset before training begins.

    For each video:
        1. Extracts N_FRAMES evenly-spaced frames.
        2. Detects face + landmarks per frame (SCRFD) -> per-frame meta.
        3. Prepares ONE consistent set of 224x224 faces for the whole video
           via prepare_video_faces (aligned if landmarks are good, else
           letterboxed — never mixed). Saved as frame_XXX.npy.
        4. Computes FFT frequency map per prepared face -> freq_XXX.npy.
        5. Computes optical flow across the prepared faces -> flow_XXX.npy.
        6. Saves the per-video decision trail -> video_meta.json.

    Skips videos that are already fully cached (safe to re-run if interrupted).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset dataframe containing at least a 'path' column with video paths.

    Returns
    -------
    None
        All outputs are saved to disk under CFG.CACHE_DIR/<video_id>/
    """

    print(f"Pre-computing cache for {len(dataframe)} videos...")
    print(f"Cache directory: {directory / 'precompute'}")

    # Track skipped (already cached) and failed videos
    skipped = 0
    failed  = 0
    done    = 0

    # Iterate over every video in the dataset (single live progress bar)
    bar = tqdm(dataframe.iterrows(), total=len(dataframe), desc="Caching")
    for idx, row in bar:

        # Get video filename without extension
        vid_id = Path(row['path']).stem

        # Build per-video cache directory path
        cache_dir = directory / "precompute" / vid_id
        cache_dir.mkdir(parents=True, exist_ok=True)   # Create folder if missing

        # ---- Check if already fully cached --------------------------------
        # Build expected file paths for all frames
        freq_paths  = [cache_dir / f'freq_{i:03d}.npy'  for i in range(CFG.N_FRAMES)]
        flow_paths  = [cache_dir / f'flow_{i:03d}.npy'  for i in range(CFG.N_FRAMES)]
        frame_paths = [cache_dir / f'frame_{i:03d}.npy' for i in range(CFG.N_FRAMES)]
        meta_path   = cache_dir / 'video_meta.json'

        # If all expected files exist, skip this video entirely
        all_cached = (
            all(_valid_npy(p) for p in freq_paths + flow_paths + frame_paths)
            and _valid_json(meta_path)
        )
        
        if all_cached:
            skipped += 1
            bar.set_postfix(done=done, skipped=skipped, failed=failed)   # live status
            continue

        try:
            # ---- Step 1: Extract raw frames from video --------------------
            # Sample N_FRAMES evenly spaced frames from the video file.
            raw_frames = sample_frames_from_video(row['path'])

            # ---- Step 2: Per-frame detection (SCRFD) ----------------------
            # Each call returns ONLY meta (bbox, landmarks, kps_ok, ...).
            metas = [extract_face_or_resize(f) for f in raw_frames]

            # ---- Step 3: Build ONE consistent set of faces for the video --
            # prepare_video_faces decides align-or-letterbox for the whole
            # video, interpolates short landmark gaps, and returns the final
            # 224x224 faces plus the decision trail.
            faces, kps_list, video_meta = prepare_video_faces(raw_frames, metas)

            # Cache the prepared RGB faces (the RGB stream).
            for i, f in enumerate(faces):
                np.save(str(cache_dir / f'frame_{i:03d}.npy'), f)

            # ---- Step 4: Compute + cache FFT frequency maps ---------------
            # Texture/frequency artifacts that distinguish real vs fake faces.
            for i, f in enumerate(faces):
                freq_map = compute_freq_map(f)                          # (H, W, 3) uint8
                np.save(str(cache_dir / f'freq_{i:03d}.npy'), freq_map)

            # ---- Step 5: Compute + cache optical flow maps ----------------
            # Motion artifacts (e.g. unnatural rigidity in fake faces).
            # The faces are already prepared, so flow does NO alignment itself.
            flow_maps = compute_optical_flow_sequence(faces)           # list of (H,W,3)
            for i, fm in enumerate(flow_maps):
                np.save(str(cache_dir / f'flow_{i:03d}.npy'), fm)

            # ---- Step 6: Cache the per-video decision trail ---------------
            # Auxiliary spoof/quality signal (mode, interpolated frames, etc.).
            with open(meta_path, 'w') as fp:
                json.dump(_jsonable(video_meta), fp)

            done += 1   # count a freshly processed video

        except Exception as e:
            # Log failed videos but continue processing the rest
            # print(f"\n[WARNING] Failed to cache video '{vid_id}': {e}")
            failed += 1
            # show the failure (and a short reason) on the bar instead of printing
            bar.set_postfix(done=done, skipped=skipped, failed=failed, last_err=str(e)[:30])
            continue

        # live status: processed / skipped / failed counters update in place
        bar.set_postfix(done=done, skipped=skipped, failed=failed)

    # ---- Summary ----------------------------------------------------------
    print(f"\nCaching complete.")
    print(f"  Processed : {len(dataframe) - skipped - failed}")
    print(f"  Skipped   : {skipped} (already cached)")
    print(f"  Failed    : {failed}")


def _valid_npy(p):
    """True if the .npy file exists AND loads without error (catches corruption)."""
    try:
        np.load(p)          # actually try to read it
        return True
    except Exception:
        return False
 
 
def _valid_json(p):
    """True if the .json file exists AND parses without error (catches corruption)."""
    try:
        with open(p) as fp:
            json.load(fp)   # actually try to parse it
        return True
    except Exception:
        return False


def _jsonable(d):
    """Convert numpy types in the video_meta dict to plain Python for json.dump."""    
    # Empty dict for the cleaned values
    out = {}
    # Loop over each key/value in video_meta
    for k, v in d.items():
        # Convert numpy ints to plain Python ints (JSON can't handle numpy types)
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        # Convert numpy floats to plain floats
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        # For lists (like bad_indices), convert any numpy ints inside to plain ints
        elif isinstance(v, (list, tuple)):
            out[k] = [int(x) if isinstance(x, (np.integer,)) else x for x in v]
        # Anything else (plain strings/numbers) copied as-is
        else:
            out[k] = v
    # Return the JSON-safe dict
    return out

