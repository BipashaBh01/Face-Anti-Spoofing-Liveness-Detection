# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *



"""
FFT looks at a single frame/image.  
It converts the image from pixel space (how image looks) to frequency space (how intensity changes across space).  
It tells:
- where image is smooth (low frequency),
- where image has sharp edges / fine textures / noise (high frequency).

It helps detect screen moire patterns, print artifacts, blur differences, fake texture inconsistencies, etc.  

Key idea: “What texture patterns exist in this face image?”

What the function does: `compute_freq_map()`  

- Converts RGB frame to grayscale (Grayscale is used because FFT is mainly detecting texture/structure patterns, so brightness (intensity) is enough for that; no need of colour)
- Resizes image
- Applies 2D FFT
- Shifts low frequencies to center
- Computes frequency magnitude
- Normalizes to image range (0–255)
- Stacks into 3 channels for CNN input

Output: Frequency texture map image
"""


#=============================================================================
# MAIN FUNCTION FOR FFT
#=============================================================================

def compute_freq_map(frame_face, size=CFG.FRAME_SIZE, device=DEVICE):
    """
    Compute the FFT magnitude (frequency) map of one face crop.

    What this does, in plain terms
    ------------------------------
    A normal image shows *where* pixels are. The FFT (Fast Fourier Transform)
    instead shows *what repeating patterns* the image is made of — smooth
    regions vs. fine, busy texture.

    Expected input
    --------------
    This should be fed the FACE CROP from extract_face_or_resize (already
    letterboxed to `size` x `size`)

    Parameters
    ----------
    frame_face : np.ndarray
        Face crop in RGB format, shape (H, W, 3), uint8. Normally already
        `size` x `size` from extract_face_or_resize.
    size : int, default=CFG.FRAME_SIZE
        Kept for interface consistency with the other pipeline stages; the
        output square size. (Resizing itself is handled upstream.)
    device : torch.device or str, default=DEVICE
        Device for the FFT computation

    Returns
    -------
    np.ndarray
        Frequency magnitude map, shape (size, size, 3), uint8. All three
        channels are identical (a grayscale map stored in RGB form).

    Notes
    -----
    - The FFT runs on the GPU; only the small final result returns to CPU,
      where np.stack and the rest of the NumPy pipeline operate.
    - Normalisation is per-frame (each map scaled by its own min/max), so
      absolute frequency strength is NOT comparable across frames — only the
      *pattern* within each map. This is fine for texture-based spoof cues.
    - The 3-channel duplication does not alter the image; it is a plain copy
      so R = G = B at every pixel.
    """
    # RGB -> grayscale (FFT is 2D, so we use a single channel).
    gray = cv2.cvtColor(frame_face, cv2.COLOR_RGB2GRAY)
    # Move the grayscale image to the GPU as a float tensor.
    g = torch.from_numpy(gray).to(device).float()
    # 2D FFT on GPU -> complex tensor of shape (size, size).
    fft = torch.fft.fft2(g)
    # Shift the zero-frequency (low frequencies) to the center.
    fshift = torch.fft.fftshift(fft)
    # Magnitude (strength of each frequency); log1p compresses the huge range.
    mag = torch.log1p(torch.abs(fshift))
    # Min-max normalize to 0-255 on GPU, then bring back to CPU as uint8.
    mn, mx = mag.min(), mag.max()
    mag = (mag - mn) / (mx - mn + 1e-8) * 255.0          # avoid divide-by-zero
    mag = mag.to(torch.uint8).cpu().numpy()              # (size, size)
    # Duplicate the single channel to 3 (RGB-like) for downstream consistency.
    return np.stack([mag, mag, mag], axis=2)             # (size, size, 3)