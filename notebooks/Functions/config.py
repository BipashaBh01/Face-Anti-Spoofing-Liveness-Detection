# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *

#=======================================================
# Detect Device
#=======================================================
# Automatic device detection
RUNTIME_MODE = "auto"        # Options: "auto", "cpu", "gpu", "mps"
# auto — pick the best available device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")                 # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")                  # Apple Silicon GPU
else:
    DEVICE = torch.device("cpu")                  # Fallback to CPU


#========================================================
# MAIN CONFIG CLASS
#========================================================
# The configuration class with all the values

class Config:
    
    # Video / Frame settings
    N_FRAMES    = 16                       # Number of frames per video
    FRAME_SIZE  = 224                      # Resize frame to 224x224

    # Basic face-crop settings
    SCRFD_MARGIN = 30                      # Add 30 pixels around detected face box
    MIN_FACE_SIZE = 80                     # Ignore faces smaller than 80 x 80 pixels
    SCRFD_DET_THRESH = 0.6                 # Minimum confidence score for accepting a detected face
    SCRFD_DET_SIZE = (640, 640)            # Detector input resolution: before searching, resize the frame to 640×640 pixels, then look for faces in it
    
    ALIGN_MIN_FRAC = 0.7                  # whole-video align/letterbox threshold: ~70%
    MAX_INTERP_GAP = 3                    # interpolation gap limit: fill gaps up to ~3 consecutive bad frames


    # Landmark-validation thresholds for valid_5pt_landmarks()
    LMK_BOX_PAD_RATIO        = 0.15      # slack around the face box for landmark-inside check
    LMK_MIN_EYE_DX_RATIO     = 0.12      # min eye separation as a fraction of box width
    LMK_MAX_EYE_DX_RATIO     = 0.75      # max eye separation as a fraction of box width
    LMK_MIN_MOUTH_DX_RATIO   = 0.08      # min mouth width as a fraction of box width
    LMK_MAX_EYE_ROLL_DEG     = 25        # max allowed eye-line tilt (degrees)
    LMK_MAX_MOUTH_ROLL_DEG   = 30        # max allowed mouth-line tilt (degrees)
    LMK_MIN_EYE_TO_MOUTH_RATIO = 0.18    # min eye->mouth vertical span / box height
    LMK_MAX_EYE_TO_MOUTH_RATIO = 0.85    # max eye->mouth vertical span / box height
    LMK_MAX_NOSE_OFFSET_RATIO  = 0.45    # max nose horizontal offset / eye distance

    # Optical flow
    FLOW_MAX = 10.0   # maximum amount of motion (in pixels) that expects a pixel to move between two consecutive frames

    # Multi-stream inputs (extra features)
    USE_FREQ_STREAM  = True                # Use frequency (FFT) features
    USE_FLOW_STREAM  = True                # Use motion (optical flow between frames

    # Model settings
    CNN_BACKBONE_RGB  = 'convnext_tiny'  # Feature extractor model
    CNN_BACKBONE_FFT  = 'efficientnet_b0'
    CNN_BACKBONE_FLOW = 'resnet34'
    CNN_FEAT_DIM    = 1280               # Output size of CNN
    PROJ_DIM        = 256                # Projection layer size
    N_STREAMS       = 3
    RNN_HIDDEN      = 256                # RNN hidden units
    RNN_LAYERS      = 2                  # Number of RNN layers
    N_HEADS         = 8                  # Transformer attention heads
    TF_LAYERS       = 2                  # Transformer layers
    TF_FF_DIM       = 512                # Feedforward layer size
    DROPOUT         = 0.3                # Dropout for regularization

    # Loss weights (how much each loss matters)
    LAMBDA_FOCAL    = 1.0                # Main classification loss
    LAMBDA_SUPCON   = 0.3                # Contrastive learning loss

    # Training settings
    BATCH_SIZE      = 32        # Number of samples per batch
    N_EPOCHS        = 30       # Number of training iterations
    LR              = 3e-4     # Learning rate
    WEIGHT_DECAY    = 1e-4     # Regularization to avoid overfitting
    GRAD_CLIP       = 1.0      # Limit gradient size (stability)
    WARMUP_EPOCHS   = 3        # Slowly increase learning at start

    # Data split ratios
    TRAIN_RATIO = 0.70   # 70% data for training
    VAL_RATIO   = 0.15   # 15% for validation
    TEST_RATIO  = 0.15   # 15% for testing

    CLASSES = {0: 'Real', 1: 'Fake'}  # Label mapping



# Create config object
CFG = Config()