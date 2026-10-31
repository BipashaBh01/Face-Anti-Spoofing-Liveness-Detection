# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *



#=========================================================
# MAIN CLASS FOR CNN
#=========================================================

class CNNFeatureExtractor(nn.Module):
    """
    CNNFeatureExtractor

    This module extracts frame-level features from a sequence of video frames using a pretrained CNN backbone.
    
    Key functionalities:
    - Uses a pretrained CNN (via timm) to extract visual features from each frame
    - Optionally freezes early layers for transfer learning
    - Applies a projection head to reduce feature dimensionality
    - Returns features in (B, T, D) format for temporal models (LSTM/Transformer)

    Args:
        backbone (str): Name of CNN architecture (e.g., 'resnet50', 'efficientnet_b0')
        out_dim (int): Output feature dimension after projection (e.g., 256)
        pretrained (bool): Whether to use pretrained ImageNet weights
        freeze_stages (int): Number of early CNN blocks to freeze (for transfer learning)

    Input:
        x (Tensor): Shape (B, T, C, H, W)
            B = batch size
            T = number of frames
            C = channels (usually 3 for RGB)
            H, W = image height and width

    Output:
        Tensor: Shape (B, T, out_dim)
            Feature vector per frame
    """

    def __init__(self, backbone, out_dim=CFG.PROJ_DIM, pretrained=True, freeze_stages=3):
        super().__init__()       # Initialises the parent PyTorch class
        # Constructor (Initialisation):
        # - backbone: which CNN model to use (RestNet, EfficientNet,etc) 
        # - out_dim: Final feature size (e.g. 256)
        # - pretrained=True:  Use weights trained on ImageNet 
        # - freeze_stages=3: freeze early layers


        # Create CNN backbone model using timm
        # - num_classes=0: removes classification head (we only need features)
        # - global_pool='avg': applies global average pooling (output is a feature vector)
        # Output: (batch_size, feature_dim)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        # Gets the size of extracted features from the CNN
        feat_dim = self.backbone.num_features

        
        # ------------------- Freeze Early Layers (Transfer Learning) -------------------
        # Freeze initial layers to retain general visual features (edges, textures)
        # Only deeper layers are fine-tuned for the specific task
        
        if freeze_stages > 0:  # Only freeze if user wants
            # Loop over every parameter (weight) in the backbone
            for name, param in self.backbone.named_parameters():
                # Split the parameter name into parts (e.g. 'stages.2.blocks...' -> ['stages','2','blocks',...])
                parts = name.split('.')
                # ---- Work out which numbered stage this parameter belongs to ----
                # Different backbones name their stages differently, so we check each:
                # ConvNeXt ('stages.N...') and EfficientNet ('blocks.N...')
                # Check the prefix matches AND the next part is a number before converting
                if parts[0] in ('stages', 'blocks') and len(parts) > 1 and parts[1].isdigit():
                    stage_idx = int(parts[1])              # e.g. 'stages.2' -> 2
                # ResNet ('layerN...', 1-indexed) -> subtract 1 to make it 0-indexed
                elif parts[0].startswith('layer') and parts[0][5:].isdigit():
                    stage_idx = int(parts[0][5:]) - 1      # e.g. 'layer3' -> 2
                # Stem / patch-embed / head / anything else -> not a numbered stage
                else:
                    stage_idx = None
    
                # ---- Decide: freeze or train this parameter ----
                if stage_idx is None:
                    # Stem/head (the very first/last bits) -> freeze
                    param.requires_grad = False
                else:
                    # Inside a numbered stage:
                    #   early stage (index < freeze_stages) -> freeze (False)
                    #   deep  stage (index >= freeze_stages) -> train  (True)
                    param.requires_grad = (stage_idx >= freeze_stages)

        # ------------------- Projection Head -------------------
        # feature refinement + dimensionality reduction
        # Transforms CNN output into desired feature space
        # Typical flow: Linear -> Normalisation -> Activation -> Regularisation
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, out_dim),            # Linear: Reduce dimension (e.g., 2048 -> 256)
            nn.BatchNorm1d(out_dim),                 # Normalises the features in a batch: mean = ~0, Var = ~1
            nn.GELU(),                               # Non-linear activation (smooth alternative to ReLU)
            nn.Dropout(CFG.DROPOUT * 0.5)            # Regularisation: Reduce overfitting --> Randomly turn off (drop) 50% neurons in each forward pass
        )

    #-------- Defines how data flows ---------------------------
    def forward(self, x):
        """
        Forward pass

        Steps:
        1. Reshape video into individual frames
        2. Pass each frame through CNN backbone
        3. Apply projection head
        4. Reshape back to sequence format

        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            Tensor: Feature tensor of shape (B, T, out_dim)
        """

        # Extract dimensions
        # Input shape:
        # B = batch size
        # T = number of frames
        # C,H,W = image channels and size
        # (Input is video frames)
        B, T, C, H, W = x.shape

        # ------------------- Frame-wise Feature Extraction -------------------
        # Reshape video to treat all frames as independently images through CNN
        # (B, T, C, H, W) --> (B*T, C, H, W)
        feats = self.backbone(x.view(B * T, C, H, W))

        # ------------------- Projection -------------------
        # Transform features to lower-dimensional embedding space
        feats = self.projector(feats)

        # ------------------- Reshape Back to Sequence -------------------
        # Now each frame has a feature vector
        # (B*T, D) --> (B, T, D)
        return feats.view(B, T, -1)