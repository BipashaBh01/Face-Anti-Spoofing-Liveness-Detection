# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *
from Functions.CNN_9 import *
from Functions.stream_gates_10 import *
from Functions.biLSTM_11 import *
from Functions.transformer_12 import *



#=========================================================
# MAIN CLASS FOR Full Model Pipeline
#=========================================================

#--------------- Baseline hybrid model (RGB + FFT + Flow) ----------------------
class HybridPADModel(nn.Module):
    """
    Baseline multi-stream Face Anti-Spoofing (FAS) model.

    Architecture Overview:
    ----------------------
    Combines spatial, temporal, and multi-stream cues using:

    1. Three CNN feature extractors (multi-stream input):
       - RGB stream  (appearance)          -> ConvNeXt-Tiny (pretrained)
       - Frequency stream (FFT texture)    -> EfficientNet-B0 (from scratch)
       - Optical Flow stream (motion)      -> ResNet-34 (from scratch)

    2. Stream Gating:
       - Learns importance weights for each stream dynamically.

    3. Temporal Modeling (parallel):
       - BiLSTM      : local/sequential temporal patterns.
       - Transformer : global temporal relationships.

    4. Fusion + Classification:
       - Concatenates BiLSTM + Transformer outputs, predicts real vs fake.

    5. Projection Head (kept, but only used if SupCon loss is added later):
       - Maps embeddings to a 128-D normalised space for contrastive learning.
       - Harmless if unused; lets SupCon be switched on later with no rebuild.

    NOTE: The auxiliary depth head has been removed for the baseline. It can be
    re-added later as a fine-tuning enhancement (needs MiDaS pseudo-labels).

    Outputs:
    --------
    logits : Raw prediction scores (B,)
    probs  : Sigmoid probabilities (B,)
    proj   : L2-normalised embeddings for contrastive learning (B, 128)
    """

    def __init__(self):
        super().__init__()

        # -------- CNN feature extractors for each modality ----------------------
        # RGB uses pretrained weights (natural images -> pretraining helps most).
        # Frequency and Flow train from scratch (non-natural inputs).
        self.cnn_rgb  = CNNFeatureExtractor(backbone='convnext_tiny',   pretrained=True,  freeze_stages=3)
        self.cnn_freq = CNNFeatureExtractor(backbone='efficientnet_b0', pretrained=False, freeze_stages=0)
        self.cnn_flow = CNNFeatureExtractor(backbone='resnet34',        pretrained=False, freeze_stages=0)

        # ------ Stream gating module -------------------------------------------
        # Learns soft attention weights across RGB, Freq, Flow streams.
        self.stream_gate = StreamGating(n_streams=CFG.N_STREAMS, dim=CFG.PROJ_DIM)

        # -------- Temporal encoders --------------------------------------------
        self.rnn_encoder = BiLSTMEncoder()       # local/sequential temporal patterns
        self.tf_encoder  = TransformerEncoder()  # global temporal dependencies

        # ------- Fusion classifier ---------------------------------------------
        # Input size = concatenation of BiLSTM and Transformer outputs.
        fusion_in = self.rnn_encoder.out_dim + self.tf_encoder.out_dim  # e.g. 512 + 256 = 768

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_in),                # stabilises training
            nn.Linear(fusion_in, 256),              # shrink to 256 numbers (first compression)
            nn.GELU(),                              # smooth non-linearity
            nn.Dropout(CFG.DROPOUT),

            # Shrink further 256 --> 64, another GELU, and a lighter dropout (half the rate). Gradually narrowing toward the decision.
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(CFG.DROPOUT * 0.5),

            # Final layer: 64 --> 1 number. This single value is the raw real/fake score (the "logit")
            nn.Linear(64, 1)                        # binary classification output
        )

        #-------- Projection head (for SupCon loss, optional) -------------------
        # Build the projection head—a separate small branch that maps the 768-number summary into a 128-number embedding, 
        # used for SupCon contrastive loss. 768 → 256 → ReLU → 128. (Only matters when SupCon is active.)
        self.proj_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Initialise custom weights for classifier & projection head
        self._init_weights()

    def _init_weights(self):
        """
        Xavier-uniform init for the Linear layers of the classifier and projection
        head; biases zeroed. (Backbones keep their own pretrained/default init.)
        """
        # Loop over the classifier and projection head (only these two—the backbones keep their own pretrained weights)
        for m in [self.classifier, self.proj_head]:
            for layer in m.modules():
                # Look at every layer inside them; act only on the Linear layers.
                if isinstance(layer, nn.Linear):
                    # Set each Linear layer's weights using Xavier initialisation,
                    # good starting values that keep gradients stable early in training.
                    nn.init.xavier_uniform_(layer.weight)
                    # Set the bias terms to zero (a clean, standard starting point)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, rgb, freq=None, flow=None):
        """
        Inputs:
        -------
        rgb  : Tensor (B, T, 3, H, W)            RGB video frames
        freq : Tensor (B, T, 3, H, W) or None    FFT maps (fallback to RGB feats if None)
        flow : Tensor (B, T, 3, H, W) or None    Optical-flow maps (fallback if None)

        Returns:
        --------
        logits : Tensor (B,)      raw classification scores
        probs  : Tensor (B,)      sigmoid probabilities
        proj   : Tensor (B, 128)  L2-normalised embeddings (for optional SupCon)
        """
        # ------- Step 1: Extract features from each stream ----------------------
        rgb_feats  = self.cnn_rgb(rgb)                                       # (B, T, D)

        # If freq/flow not provided, reuse RGB features (graceful fallback).
        freq_feats = self.cnn_freq(freq) if freq is not None else rgb_feats # (B, T, D)
        flow_feats = self.cnn_flow(flow) if flow is not None else rgb_feats # (B, T, D)

        # ----------- Step 2: Fuse streams using learned gating ------------------
        fused_feats = self.stream_gate([rgb_feats, freq_feats, flow_feats]) # (B, T, D)

        # ------------ Step 3: Temporal encoding ---------------------------------
        rnn_out = self.rnn_encoder(fused_feats)  # (B, 2*hidden)
        tf_out  = self.tf_encoder(fused_feats)   # (B, d_model)

        # ------ Step 4: Combine temporal features -------------------------------
        embedding = torch.cat([rnn_out, tf_out], dim=1)  # (B, fusion_in)

        # --------- Step 5: Classification ---------------------------------------
        logits = self.classifier(embedding).squeeze(-1)  # (B,)
        probs  = torch.sigmoid(logits)                   # (B,)

        # ----------- Step 6: Projection (for optional contrastive learning) -----
        proj = F.normalize(self.proj_head(embedding), dim=1)  # (B, 128)

        # ------ Final outputs ---------------------------------------------------
        return logits, probs, proj
