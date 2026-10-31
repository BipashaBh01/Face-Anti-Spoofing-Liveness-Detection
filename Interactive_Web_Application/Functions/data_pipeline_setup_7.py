# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *
from Functions.precompute_5 import *
from Functions.FFT_3 import *
from Functions.optical_flow_4 import *



"""
The below class:
- Reads videos from dataset
- Extracts frames
- Creates:
  - RGB frames (appearance)
  - FFT frequency maps (texture clues)
  - Optical flow maps (motion clues)
- Converts them into tensors
"""

#=========================================================
# MAIN CLASS FOR DATA PIPLEINE SETUP
#=========================================================

class PADVideoDataset(Dataset):
    """
    Multi-stream dataset for face anti-spoofing.

    Each item is one video, returned as three aligned streams plus labels:
        rgb         : (T, 3, H, W)  RGB face frames
        freq        : (T, 3, H, W)  FFT texture maps
        flow        : (T, 3, H, W)  optical-flow motion maps
        label       : int           0 = real, 1 = fake
        attack_type : str           spoof category (print / replay / mask / ...)
        usable_frac : float         per-video quality signal from prepare stage
                                    (fraction of frames with usable landmarks)

    All three streams come from the same prepared faces (aligned-or-letterboxed
    per video, decided upstream), so they correspond spatially for gated fusion.

    Caching
    -------
    Streams are read from the precompute cache (frame/freq/flow .npy + the
    video_meta.json quality signal) for speed. If the cache is missing, the
    item is rebuilt on the fly via the same pipeline used at precompute time
    (slower, but never crashes).

    Augmentation
    ------------
    RGB augmentation (`transform`) is applied to the RGB stream ONLY. The freq
    and flow streams are converted straight to tensors with NO augmentation,
    since RGB-style photometric/geometric augments would corrupt their
    frequency/motion channels.
    """

    def __init__(self, dataframe, real_dir, cache_dir, transform=None,
                 demo_mode=True, use_freq=True, use_flow=True):
        """
        Parameters
        ----------
        dataframe : pd.DataFrame
            Rows with at least 'path' and 'label' (optional 'attack_type').
        transform : albumentations.Compose or None
            RGB-stream preprocessing/augmentation. Applied to RGB ONLY.
        demo_mode : bool
            Use synthetic frames (for pipeline testing without real data).
        use_freq, use_flow : bool
            Enable/disable the FFT / flow streams.
        """
        # Store the dataframe
        self.df = dataframe.reset_index(drop=True)
        self.real_dir = real_dir
        self.cache_dir = cache_dir
        # Store the RGB augmentation/preprocessing
        self.transform = transform
        # Turn on demo mode if explicitly requested or if the real-data folder doesn't exist (auto-fallback to synthetic data)
        self.demo_mode = demo_mode or not self.real_dir.exists()
        # Store whether to produce the FFT and flow streams
        self.use_freq = use_freq
        self.use_flow = use_flow

    def __len__(self):
        # Tells PyTorch how many samples exist (number of videos)
        return len(self.df)

    # -------------------------------------------------------------------------
    # Create synthetic data
    def _generate_demo_frames(self, label):
        """Create synthetic skin-toned frames for testing without real data."""
        frames = []
        base = 20 if label == 0 else 0                 # add a slight brightness offset for real (label 0) data
        # Loop to create N frames
        for t in range(CFG.N_FRAMES):
            # Add random noise per frame (more for real, less for fake) to make synthetic data look varied
            jitter = np.random.randn(CFG.FRAME_SIZE, CFG.FRAME_SIZE, 3) * (8 if label == 0 else 2)
            # Build a skin-toned frame: a constant color [180+base+t, 130, 110] (brightness drifts over time via t), 
            # plus the jitter, clipped to 0–255 and made uint8
            frame = np.clip(
                np.ones((CFG.FRAME_SIZE, CFG.FRAME_SIZE, 3), dtype=np.float32)
                * [180 + base + t, 130, 110] + jitter, 0, 255
            ).astype(np.uint8)
            # Collect the synthetic frames
            frames.append(frame)
        # Return the synthetic frames
        return frames

    # -------------------------------------------------------------------------
    # load or rebuild real faces
    def _prepare_faces(self, video_path, cache_dir):
        """
        Return the video's prepared 224x224 faces.

        Loads cached frame_XXX.npy if present; otherwise rebuilds them with the
        full pipeline: sample -> per-frame detect -> prepare_video_faces (the
        same consistent align-or-letterbox decision used at precompute time).
        """
        # Get the video's id and build the expected cached frame paths
        vid_id = Path(video_path).stem
        fps = [cache_dir / "precompute" / vid_id / f'frame_{i:03d}.npy' for i in range(CFG.N_FRAMES)]

        # Fast path: load cached prepared faces.
        if all(p.exists() for p in fps):
            return [np.load(str(p)) for p in fps]

        # Fallback: rebuild from the raw video via the current pipeline.
        raw = sample_frames_from_video(video_path)
        metas = [extract_face_or_resize(f) for f in raw]
        faces, _kps, _meta = prepare_video_faces(raw, metas)
        return faces

    # -------------------------------------------------------------------------
    def _to_tensor(self, img, tfm=None):
        """Apply a transform if given, else plain HWC->CHW float[0,1] tensor."""
        if tfm:
            return tfm(image=img)['image']
        return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255

    # -------------------------------------------------------------------------
    def _load_usable_frac(self, cache_dir):
        """Read the per-video quality signal (usable_frac) from video_meta.json."""
        # Open the video's video_meta.json, parse it, return usable_frac. 
        # If the file is missing/corrupt, return 0.0 safely instead of crashing
        meta_path = cache_dir / 'video_meta.json'
        try:
            with open(meta_path) as fp:
                return float(json.load(fp).get('usable_frac', 0.0))
        except Exception:
            return 0.0   # cache missing / unreadable -> neutral default

    # -------------------------------------------------------------------------
    # Load one sample
    def __getitem__(self, idx):
        """Load one video as (rgb, freq, flow, label, attack_type, usable_frac)."""
        # PyTorch calls this to fetch one video by index
        # Get the row, its label (0/1), and attack type (default 'unknown' if absent)
        row = self.df.iloc[idx]
        label = int(row['label'])
        attack_type = row.get('attack_type', 'unknown')
       
        # Build this video's cache folder path (for loading freq/flow/meta)
        vid_id    = Path(row['path']).stem
        # cache_dir_inside = self.cache_dir / 'precompute' / vid_id
        row_cache_dir = row['cache_dir'] if 'cache_dir' in row else self.cache_dir       
        cache_dir_inside = row_cache_dir / 'precompute' / vid_id

        # Get frames: synthetic if demo mode, else the prepared real faces..
        # frames = (self._generate_demo_frames(label)
        #          if self.demo_mode else self._prepare_faces(row['path']))
        frames = self._generate_demo_frames(label) if self.demo_mode else self._prepare_faces(row['path'], row_cache_dir)
               
        # ---- Stream 1: RGB (augmented) --------------------------------------
        # Apply the RGB transform to each frame, convert to tensors, stack into shape (T, 3, H, W)
        rgb = torch.stack([self._to_tensor(f, self.transform) for f in frames])

        # ---- Stream 2: FFT (no augmentation) --------------------------------
        # If enabled: for each frame, load the cached freq map (or recompute if missing), 
        # convert to tensor with no transform (no augmentation), stack. 
        # If disabled: return a zero placeholder matching rgb's shape
        if self.use_freq:
            freq_tensors = []
            for i in range(CFG.N_FRAMES):
                fp = cache_dir_inside / f'freq_{i:03d}.npy'
                freq_f = np.load(str(fp)) if fp.exists() else compute_freq_map(frames[i])
                freq_tensors.append(self._to_tensor(freq_f))      # plain tensor
            freq = torch.stack(freq_tensors)
        else:
            freq = torch.zeros_like(rgb)

        # ---- Stream 3: Optical flow (no augmentation) -----------------------
        # If enabled: if all flow files exist, load them; 
        # otherwise recompute the whole sequence once (flow needs the full sequence). 
        # Convert each to tensor with no transform, stack. If disabled: zero placeholder.
        if self.use_flow:
            # Recompute once only if any flow file is missing (flow is a sequence).
            flow_cached = [cache_dir_inside / f'flow_{i:03d}.npy' for i in range(CFG.N_FRAMES)]
            if all(p.exists() for p in flow_cached):
                flow_maps = [np.load(str(p)) for p in flow_cached]
            else:
                flow_maps = compute_optical_flow_sequence(frames)
            flow = torch.stack([self._to_tensor(fm) for fm in flow_maps])  # plain tensor
        else:
            flow = torch.zeros_like(rgb)

        # Per-video quality signal (0.0 in demo mode where there's no meta).
        usable_frac = 0.0 if self.demo_mode else self._load_usable_frac(cache_dir_inside)

        return rgb, freq, flow, label, attack_type, usable_frac


