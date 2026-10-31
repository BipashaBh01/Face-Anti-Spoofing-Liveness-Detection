# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


"""
`get_transforms()`: The function that defines image augmentation pipelines for training and validation.  
- Training: Applies random transformations (flip, blur, brightness, noise, compression, etc.)
- Validation: Only resizes and normalizes (no randomness)

Why it is needed:
- Prevents overfitting
- Makes model robust to lighting changes, blur, camera quality, occlusions.
- Ensures fair evaluation during validation.

Pipeline:  
**For training:** resize images, flip horizontally, rotate / zoom / shift, add blur, change brightness / contrast / colour, add occlusion (dropout), add noise, simulate JPEG compression, normalize.
Then convert to tensor.  
**For validation/test:** Resize, normalise. Then convert to tensor.
"""

#=============================================================================
# MAIN FUNCTION FOR AUGMENTATION STEP
#=============================================================================

def get_transforms(mode='train'):
    """
    Build the image transform pipeline for the RGB stream.
 
    What this does
    ------------------------------
    Returns an Albumentations transform pipeline that prepares an image before
    it goes into the model. Two jobs:
      1. Normalise (both modes): rescale pixels to the ImageNet mean/std the
         pretrained CNN backbone expects, so it sees inputs in the range it was
         trained on and trains stably.
      2. Augment (train only): apply random distortions so the model sees varied
         versions of each face across epochs and generalises instead of
         memorising. Validation/test skip this for fair, repeatable evaluation.
 
    Augmentation is applied in memory at load time (not saved to disk), so the
    same clean cached frame yields a different random result each epoch.
 
    Intended scope
    --------------
    These augmentations are designed for the RGB STREAM. Several of them (flip,
    blur, noise, colour shifts, compression) would corrupt the FFT and optical-
    flow streams, whose channels encode frequency/motion rather than colour.
    Therefore, this pipeline is applied to RGB only.
 
    Parameters
    ----------
    mode : str, default='train'
        'train' -> resize + full random augmentation + normalize + to-tensor.
        anything else (e.g. 'val'/'test') -> resize + normalize + to-tensor only
        (deterministic, no randomness).
 
    Returns
    -------
    albumentations.Compose
        A callable transform. Applied as: out = transform(image=img)['image'],
        which returns a normalized CHW float tensor (via ToTensorV2).
 
    Transform list (train mode)
    ---------------------------
    - Resize(FRAME_SIZE)        : force a fixed square size (safety; inputs are
                                  usually already 224x224).
    - HorizontalFlip(p=0.5)     : 50% chance to mirror left/right.
    - Affine(p=0.4)             : random small shift (5%), zoom (0.9-1.1x),
                                  rotation (+/-10 deg) -> pose/camera robustness.
    - OneOf[Gaussian/Motion blur](p=0.3) : 30% chance to apply ONE blur type
                                  -> low-quality / motion-blurred frames.
    - RandomBrightnessContrast(p=0.5)    : vary brightness & contrast (+/-25%)
                                  -> lighting robustness.
    - HueSaturationValue(p=0.3) : shift colour tone / intensity / brightness
                                  -> camera colour variation.
    - CoarseDropout(p=0.3)      : black out 1-8 small rectangles -> occlusion
                                  robustness (don't rely on one face region).
    - GaussNoise(p=0.25)        : add sensor/compression-like grain.
    - ImageCompression(p=0.2)   : simulate JPEG artifacts -> real-world quality,
                                  and reduce a compression-based shortcut.
    - Normalize(mean,std)       : ImageNet normalization.
    - ToTensorV2()              : HWC uint8 -> CHW float tensor for PyTorch.
 
    Notes
    -----
    - mean/std are the standard ImageNet statistics; required when using an
      ImageNet-pretrained backbone (inputs must match its training range).
    - Each augmentation has its own probability `p`, so a given frame gets a
      random subset applied — sometimes almost nothing — which is the source of
      cross-epoch variety.
    """
    # Mean and standard deviation used for ImageNet-style normalisation
    # Helps scale pixel values so model trains more stably
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Training augmentations: add randomness to improve generalisation
    if mode == 'train':
        return A.Compose([

            A.Resize(CFG.FRAME_SIZE, CFG.FRAME_SIZE),

            # (removed HorizontalFlip and Affine — they move the RGB face but the
            #  precomputed FFT/flow use the un-moved frame, desyncing the streams)

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
            ], p=0.3),

            A.RandomBrightnessContrast(          # dim-light bias -> BPCER
                brightness_limit=(-0.30, 0.15),
                contrast_limit=0.25,
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(70, 150), p=0.3),   # dim light

            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3
            ),

            A.CoarseDropout(                      # glasses / occlusion
                num_holes_range=(1, 8),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=0, p=0.35
            ),

            A.GaussNoise(std_range=(0.02, 0.1), p=0.25),

            A.ImageCompression(quality_range=(40, 100), p=0.3),  # unseen-device quality

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # Validation / test transforms: only resize + normalize
    # No random changes, to ensure fair evaluation
    else:
        return A.Compose([

            # Resize input to fixed model size
            A.Resize(CFG.FRAME_SIZE, CFG.FRAME_SIZE),

            # Normalize pixel values
            A.Normalize(mean=mean, std=std),

            # Convert to tensor
            ToTensorV2(),
        ])
