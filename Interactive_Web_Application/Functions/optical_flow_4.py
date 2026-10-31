# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *

"""
Optical flow (inter-frame) -  motion analysis

Optical flow compares two consecutive frames.  
It estimates how each pixel moves between frames.  
It tells: direction of movement, speed/magnitude of movement.  

It helps detect natural eye blinks, subtle head motion, facial muscle movement, rigid movement in replay attacks, etc.  

Key idea: “How is the face moving over time?”

What the function does: `compute_optical_flow_sequence()`
- Converts all frames to grayscale, then resize
- For each frame pair: computes optical flow (motion)
- Extracts: x-direction motion, y-direction motion, motion magnitude
- Normalizes all channels
- Combines into 3-channel motion map

Output: Sequence of motion maps
"""


#==================================================
# Load RAFT-Large
#==================================================

# Load the pretrained RAFT-Large weights.
_weights = Raft_Large_Weights.DEFAULT

# Build the model, eval mode (inference), move to your device.
RAFT_MODEL = raft_large(weights=_weights).eval().to(DEVICE)

# Official preprocessing that ships with these weights.
RAFT_TRANSFORMS = _weights.transforms()



#=============================================================================
# MAIN FUNCTION FOR OPTICAL FLOW
#=============================================================================

# RAFT-Large optical flow over a sequence of face frames
def compute_optical_flow_sequence(
    frames_rgb,
    size=CFG.FRAME_SIZE,
    model=RAFT_MODEL,
    transforms=RAFT_TRANSFORMS,
    device=DEVICE,
    batch_size=4,
    flow_max=CFG.FLOW_MAX,
):
    """
    Compute motion between consecutive face frames using RAFT-Large on GPU.

    What this does
    ------------------------------
    Optical flow measures how each part of the image MOVES from one frame to
    the next. For face anti-spoofing this matters because a live face moves in
    natural 3D ways (blinks, small head turns, muscle motion), while a fake
    moves differently — a printed photo shifts rigidly as one flat sheet, and a
    screen replay can look jittery or flat. The motion pattern is a spoof clue.
    RAFT is a deep-learning flow model; "Large" is the more accurate variant,
    chosen because the subtle facial motion that signals liveness is exactly
    what smaller / classical methods tend to miss.

    Design choice
    ----------------------
    GLOBAL NORMALISATION: motion is scaled by ONE fixed range (flow_max, in
    pixels) for every frame, so motion magnitude is COMPARABLE across frames
    (a low-motion frame stays dim, a high-motion frame stays bright). This
    replaces per-frame min/max scaling, which destroyed that comparison.

    Expected input
    --------------
    The faces from prepare_video_faces — each one is a
    `size` x `size` face that the whole-video stage has consistently aligned OR
    letterboxed. Alignment is therefore NOT done here (doing it again would be a
    double warp). The sequence must stay in true temporal order with no gaps,
    since RAFT pairs adjacent frames; filter whole videos, never single frames.

    Pipeline (step by step)
    -----------------------
    1. Convert each prepared face to a float tensor in [0, 1], shape (3, H, W).
    2. Stack to (N, 3, H, W) and form consecutive pairs: frames[:-1] vs frames[1:].
    3. Run RAFT on the GPU in batches; use its final, most-refined flow.
    4. For each flow field, compute magnitude and apply FIXED global scaling to
       x / y / magnitude, packing them into a 3-channel uint8 motion map.
    5. Duplicate the last map once so output length == number of input frames.

    Parameters
    ----------
    frames_rgb : list of np.ndarray
        Consecutive ALREADY-PREPARED face images, each (size, size, 3) uint8,
        from prepare_video_faces.
    size : int, default=CFG.FRAME_SIZE
        Square size for RAFT input. Must be divisible by 8 (RAFT requirement).
    model : RAFT model, default=RAFT_MODEL
        Pre-loaded RAFT-Large network (already on `device`, in eval mode).
    transforms : callable, default=RAFT_TRANSFORMS
        Official RAFT preprocessing transform shipped with the weights.
    device : torch.device or str, default=DEVICE
        Project-wide GPU/CPU device.
    batch_size : int, default=4
        Frame pairs processed per GPU batch. Lower this first if you hit
        out-of-memory with RAFT-Large.
    flow_max : float, default=CFG.FLOW_MAX
        Fixed maximum motion (in pixels) used for normalization. Pick it from
        the typical motion in your faces (e.g. ~95th percentile of observed
        pixel motion). Too small -> everything saturates; too large -> looks static.

    Returns
    -------
    list of np.ndarray
        N motion maps, each (size, size, 3) uint8:
            channel 0 = x motion, channel 1 = y motion, channel 2 = magnitude.

    Notes
    -----
    - GPU-native: RAFT runs on `device`; only small results return to CPU.
    - x/y are SIGNED: 0 motion -> 128 (mid-gray), +flow_max -> 255, -flow_max -> 0.
      Magnitude is unsigned: 0..flow_max -> 0..255.
    - The final map is a duplicate (padding), not real motion.
    - Frames arrive consistently aligned/letterboxed from prepare_video_faces,
      so head-motion cancellation (when aligned) already happened upstream.
    """
    # No frames -> return an empty list
    if len(frames_rgb) == 0:
        return []
    # One frame --> no pair to compare, so return a single neutral (128 = no motion) map
    if len(frames_rgb) == 1:
        return [np.full((size, size, 3), 128, dtype=np.uint8)]
    # RAFT requires the side length divisible by 8; stop early if it is not
    if size % 8 != 0:
        raise ValueError("RAFT input size must be divisible by 8, e.g. 224, 256, 320.")

    # --- 1. Convert each face -> (3, H, W) float tensor ------------------
    # Loop over each prepared face, convert it from a (H,W,3) uint8 image into a (3,H,W) float tensor in [0,1] (TF.to_tensor). 
    # Stack them all into one tensor of shape (N, 3, H, W).
    tensors = []
    for f in frames_rgb:
        tensors.append(TF.to_tensor(f))        # HWC uint8 -> CHW float in [0,1]
    frames_tensor = torch.stack(tensors, dim=0)  # (N, 3, H, W)

    # --- 2. Build consecutive frame pairs ---------------------------------
    img1_all = frames_tensor[:-1]               # frames 0 .. N-2 -OR- img1_all = all frames except the last
    img2_all = frames_tensor[1:]                # frames 1 .. N-1 -OR- img2_all = all frames except the first

    # --- 3. Run RAFT in batches ----------------------------
    flow_maps = []
    with torch.no_grad():                                        # inference only, no gradients
        for start in range(0, img1_all.shape[0], batch_size):
            # Loop over the pairs in chunks of batch_size (4). 
            # Slice out this batch of "before" frames (img1) and "after" frames (img2)
            end = start + batch_size
            img1, img2 = img1_all[start:end], img2_all[start:end]

            img1, img2 = transforms(img1, img2)                  # Apply RAFT's preprocessing
            img1, img2 = img1.to(device), img2.to(device)        # Move to GPU    

            # Run RAFT. 
            # It returns a list of progressively refined flow predictions.
            # [-1] takes the final, best one, 
            # detach drops gradient info, 
            # cpu().numpy() brings it back as a NumPy array. Shape: (batch, 2, H, W)
            flows = model(img1, img2)[-1].detach().cpu().numpy()

            # --- 4.  Convert each flow to a 3-channel uint8 map -----
            for flow in flows:
                # Loop over each flow in the batch. Rearrange from (2,H,W) to (H,W,2) 
                # so the last axis holds the (x, y) motion per pixel
                flow = np.transpose(flow, (1, 2, 0))            # (2,H,W) -> (H,W,2)

                # Split into x-motion (fxf) and y-motion (fyf), then compute magnitude (total motion = √(x²+y²))
                fxf, fyf = flow[..., 0], flow[..., 1]           # signed x / y motion
                mag = np.sqrt(fxf**2 + fyf**2)                  # motion strength

                # Scale x and y to the signed 0–255 range: no motion → 128, +flow_max → 255, −flow_max → 0. 
                # clip keeps it in [0,255], then convert to uint8.
                fx = np.clip((fxf / flow_max) * 127.5 + 127.5, 0, 255).astype(np.uint8)
                fy = np.clip((fyf / flow_max) * 127.5 + 127.5, 0, 255).astype(np.uint8)
                # Scale magnitude to unsigned 0–255: 0 → 0, flow_max → 255
                fm = np.clip((mag / flow_max) * 255.0, 0, 255).astype(np.uint8)

                # Stack the three channels into one (H, W, 3) motion map and add it to the list
                flow_maps.append(np.stack([fx, fy, fm], axis=2))  # (H, W, 3)

    # --- 5. Pad so output length == number of input frames ----------------
    # N frames make N−1 flow maps. Duplicate the last map once so the output length equals N (one map per input frame),
    # then return the list.
    flow_maps.append(flow_maps[-1].copy())
    return flow_maps
