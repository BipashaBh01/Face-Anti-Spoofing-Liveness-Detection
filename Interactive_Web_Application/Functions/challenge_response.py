# Import all necessary packages
from lib_imports import *
from Functions.config import *
from Functions.face_detect_2 import *
from Functions.FFT_3 import *
from Functions.optical_flow_4 import *
from Functions.full_model_pipeline_13 import *



# ============================================================
# BUILD MODEL INPUTS FROM A VIDEO SOURCE (reusable)
# ============================================================

# ---- transforms + tensor helpers ----
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

rgb_transform = A.Compose([
    A.Resize(CFG.FRAME_SIZE, CFG.FRAME_SIZE),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])

def to_tensor_plain(img):
    """Mirror of Dataset._to_tensor(img, tfm=None): HWC->CHW float[0,1]."""
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255

def to_tensor_rgb(img):
    """Mirror of Dataset._to_tensor(img, self.transform) on the val branch."""
    return rgb_transform(image=img)["image"]

# ---------- Main function ------------------------
def build_model_inputs(video_path, device, verbose_prep=True):
    """
    Build the three model input tensors from a single video.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file (live or spoof).
    device : torch.device
        Device to place the output tensors on.
    verbose : bool
        If True, print per-stage shape/quality diagnostics.

    Returns
    -------
    dict with keys:
        'rgb'          : Tensor (1, T, 3, H, W)
        'freq'         : Tensor (1, T, 3, H, W)
        'flow'         : Tensor (1, T, 3, H, W)
        'landmarks'    : list of per-frame (5, 2) landmark arrays
        'prep_info'    : dict (usable_frac, n_good, n_interpolated, ...)
        'video_path'   : Path
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    # 1. Sample raw frames
    raw_frames = sample_frames_from_video(video_path)

    # 2. Detect faces + metadata per frame
    metas = [extract_face_or_resize(f) for f in raw_frames]

    # 3. Crop / align
    face_crops, face_landmarks, prep_info = prepare_video_faces(raw_frames, metas)

    # 4. Frequency maps (one call per face)
    freq_maps = [compute_freq_map(crop) for crop in face_crops]
    freq_maps = [
        (fm.detach().cpu().numpy() if torch.is_tensor(fm) else fm)
        for fm in freq_maps
    ]

    # 5. Optical flow (whole sequence at once)
    flow_maps = compute_optical_flow_sequence(face_crops)

    # 6. Sanity: all three streams == N_FRAMES
    assert len(face_crops) == len(freq_maps) == len(flow_maps) == CFG.N_FRAMES, (
        f"Stream length mismatch: rgb={len(face_crops)}, "
        f"freq={len(freq_maps)}, flow={len(flow_maps)}, expected={CFG.N_FRAMES}"
    )

    # 7. To tensors, matching training _to_tensor usage exactly
    rgb_stack  = torch.stack([to_tensor_rgb(face_crops[i])  for i in range(CFG.N_FRAMES)])
    freq_stack = torch.stack([to_tensor_plain(freq_maps[i]) for i in range(CFG.N_FRAMES)])
    flow_stack = torch.stack([to_tensor_plain(flow_maps[i]) for i in range(CFG.N_FRAMES)])

    # 8. Batch dim + device
    rgb_tensor  = rgb_stack.unsqueeze(0).to(device)
    freq_tensor = freq_stack.unsqueeze(0).to(device)
    flow_tensor = flow_stack.unsqueeze(0).to(device)

    if verbose_prep:
        print(f"[{video_path.name}]")
        print(f"  Faces found : {sum(1 for m in metas if m['face_found'])}/{len(metas)}")
        print(f"  Quality     : usable_frac={prep_info['usable_frac']:.2f}, "
              f"n_good={prep_info['n_good']}, n_interp={prep_info['n_interpolated']}, "
              f"n_unfilled={prep_info['n_unfilled']}")
        print(f"  RGB  {tuple(rgb_tensor.shape)} | "
              f"Freq {tuple(freq_tensor.shape)} | Flow {tuple(flow_tensor.shape)}")

    return {
        "rgb": rgb_tensor,
        "freq": freq_tensor,
        "flow": flow_tensor,
        "landmarks": face_landmarks,
        "prep_info": prep_info,
        "video_path": video_path,
    }



# ============================================================
# RUN INFERENCE + APPLY DECISION THRESHOLD (reusable)
# ============================================================
#
# forward() returns: (logits, probs, proj)
#   probs -> (B,) sigmoid probability of the POSITIVE class.
#
# label==1 means SPOOF, so SPOOF_IS_POSITIVE = True.
# ============================================================

def score_video(inputs, model, threshold, spoof_is_positive, verbose_score=True):
    """
    Run the FAS model on prepared inputs and return the spoof/live verdict.

    Parameters
    ----------
    inputs : dict
        Output of build_model_inputs(...) — must contain 'rgb', 'freq',
        'flow', 'prep_info', 'video_path'.
    model : nn.Module
        The trained HybridPADModel (eval mode enforced internally).
    threshold : float
        Decision threshold on spoof probability (>= threshold -> SPOOF).
    spoof_is_positive : bool
        True if the model's positive class (label==1) is SPOOF.
    verbose : bool
        Print a formatted result block.

    Returns
    -------
    dict with keys:
        'video'        : str  (filename)
        'logit'        : float
        'positive_prob': float
        'spoof_prob'   : float
        'live_prob'    : float
        'threshold'    : float
        'verdict'      : str  ('SPOOF' or 'LIVE')
        'usable_frac'  : float
    """
    model.eval()
    with torch.no_grad():
        logits, probs, proj = model(
            inputs["rgb"], inputs["freq"], inputs["flow"]
        )

    positive_prob = probs.squeeze().item()
    spoof_prob = positive_prob if spoof_is_positive else (1.0 - positive_prob)
    live_prob  = 1.0 - spoof_prob
    verdict = "SPOOF" if spoof_prob >= threshold else "LIVE"

    prep_info = inputs["prep_info"]
    result = {
        "video":         inputs["video_path"].name,
        "logit":         logits.squeeze().item(),
        "positive_prob": positive_prob,
        "spoof_prob":    spoof_prob,
        "live_prob":     live_prob,
        "threshold":     threshold,
        "verdict":       verdict,
        "usable_frac":   prep_info["usable_frac"],
    }

    if verbose_score:
        print("=" * 70)
        print("FAS INFERENCE RESULT")
        print("=" * 70)
        print(f"Video            : {result['video']}")
        print(f"Raw logit        : {result['logit']:+.4f}")
        print(f"Positive-class p : {result['positive_prob']:.4f}  "
              f"(SPOOF_IS_POSITIVE={spoof_is_positive})")
        print(f"Spoof prob       : {result['spoof_prob']:.4f}")
        print(f"Live prob        : {result['live_prob']:.4f}")
        print(f"Threshold        : {result['threshold']}")
        print(f"VERDICT          : {result['verdict']}")
        print(f"Clip quality     : usable_frac={result['usable_frac']:.2f}")
        print("=" * 70)

    return result


# ====================================================================================
# Combining BUILD + INFERENCE as a single 'check video' function
# ====================================================================================
def check_video(video_path, threshold, spoof_is_positive, model, device, verbose_prep=False, verbose_score=True):
    """
    Prepare inputs and score a video in one call.

    Parameters
    ----------
    video_path      : str or Path  — the video to check.
    verbose_prep    : bool  — print input-prep diagnostics (faces, quality, shapes).
    verbose_score   : bool  — print the FAS inference result block.
    threshold       : float — spoof decision threshold.
    spoof_is_positive : bool — True if model's positive class is SPOOF.
    model           : nn.Module — the trained model.

    Returns
    -------
    dict — the result from score_video (video, spoof_prob, verdict, ...).
    """
    inputs = build_model_inputs(
        video_path,
        device=device,
        verbose_prep=verbose_prep
    )
    return score_video(
        inputs,
        model=model,
        threshold=threshold,
        spoof_is_positive=spoof_is_positive,
        verbose_score=verbose_score,
    )
