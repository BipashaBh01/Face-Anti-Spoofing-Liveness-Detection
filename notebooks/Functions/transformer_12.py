# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


#=========================================================
# MAIN CLASS FOR Transformer
#=========================================================

class TransformerEncoder(nn.Module):
    """
    A Transformer Encoder module for sequence classification using a [CLS] token.

    Prepends a learnable [CLS] token to the input sequence, applies sinusoidal
    positional encoding, passes the sequence through a stack of Transformer encoder
    layers, and returns the final [CLS] token representation as the sequence embedding.

    Args:
        d_model  (int): Dimensionality of the model (embedding size). Defaults to CFG.PROJ_DIM.
        n_heads  (int): Number of attention heads in each encoder layer. Defaults to CFG.N_HEADS.
        n_layers (int): Number of stacked Transformer encoder layers. Defaults to CFG.TF_LAYERS.
        ff_dim   (int): Hidden dimensionality of the feed-forward sublayer. Defaults to CFG.TF_FF_DIM.

    Input:
        x (Tensor): Input tensor of shape ``(B, T, d_model)`` where
                    ``B`` is the batch size and ``T`` is the sequence length.

    Output:
        Tensor: [CLS] token embedding of shape ``(B, d_model)``,
                representing the aggregated sequence feature.
    """

    def __init__(self, d_model  = CFG.PROJ_DIM, n_heads  = CFG.N_HEADS, n_layers = CFG.TF_LAYERS, ff_dim   = CFG.TF_FF_DIM):
        # Initialises the model
        # d_model -> embedding size (feature dimension),
        # n_heads -> number of attention heads,
        # n_layers -> number of Transformer layers,
        # ff_dim -> hidden size of feed-forward network
        
        super().__init__()    # Calls parent class (nn.Module) constructor.

        # Create learnable [CLS] token (vector) 
        # shape: (1 × 1 × d_model)
        # This acts like a summary token. During forward pass --> copied for each batch.
        # This token collects information from the whole sequence.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional Encoding Setup (Maximum sequence length)
        # Total positional slots = sequence frames + 1 slot reserved for the [CLS] token
        max_len = CFG.N_FRAMES + 1

        # Create matrix to store positional encodings       
        # Allocate the sinusoidal positional encoding table (shape: max_len × d_model) or (sequence_length, embedding_dim)
        pos = torch.zeros(max_len, d_model)

        # Creates position indices: [0, 1, 2, ...]
        # Column vector of absolute positions [0, 1, …, max_len-1] (shape: max_len × 1)
        positions = torch.arange(0, max_len).float().unsqueeze(1)

        # Frequency divisors for each even dimension pair — decays geometrically with depth
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Computes scaling factors for sine/cosine (Controls frequency of waves)
        # Even dimensions get sine encodings
        pos[:, 0::2] = torch.sin(positions * div)
        # Odd dimensions get cosine encodings
        pos[:, 1::2] = torch.cos(positions * div)

        # Register as a non-trainable buffer so it moves with the model (e.g. .cuda())
        # Unsqueeze adds the batch dimension --> shape: 1 × max_len × d_model
        self.register_buffer('pos_enc', pos.unsqueeze(0))

        # Single Transformer encoder layer with Pre-LN ("norm_first") and GELU activation
        enc_layer = nn.TransformerEncoderLayer(
            d_model,                # embedding dimensionality
            n_heads,                # number of self-attention heads
            ff_dim,                 # feed-forward hidden size
            CFG.DROPOUT,            # dropout probability applied throughout the layer
            'gelu',                 # smoother activation vs. ReLU, common in modern Transformers
            batch_first = True,     # expects input as (Batch, Seq, Feature)
            norm_first  = True,     # Pre-LN: normalise before attention/FF for training stability
        )

        # Stack n_layers encoder layers with a final LayerNorm over the output
        self.tf = nn.TransformerEncoder(
            enc_layer,
            n_layers,
            norm = nn.LayerNorm(d_model),   # applied to every token after the last layer
        )

        # Dropout applied to the embedded + position-encoded input before the encoder
        self.drop = nn.Dropout(CFG.DROPOUT)

        # Expose output dimensionality so downstream modules can query it without hard-coding
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Encoder.

        Args:
            x (Tensor): Projected frame features of shape ``(B, T, d_model)``.

        Returns:
            Tensor: Aggregated sequence embedding of shape ``(B, d_model)``
                    extracted from the [CLS] token position.
        """

        # Unpack batch size B and sequence length T (d_model is implicitly validated by the layers)
        B, T, _ = x.shape

        # Expand the single [CLS] token to match the current batch size -> (B, 1, d_model)
        cls = self.cls_token.expand(B, -1, -1)

        # Prepend [CLS] to the frame sequence, then add positional encodings
        # Sequence layout after cat: [CLS, frame_0, frame_1, …, frame_{T-1}]
        # pos_enc is sliced to exactly T+1 positions to handle variable-length inputs
        x = torch.cat([cls, x], dim=1) + self.pos_enc[:, :T+1, :]

        # Apply input dropout, run through all encoder layers, then return only the
        # [CLS] token (index 0) as the fixed-size sequence representation --> (B, d_model)
        return self.tf(self.drop(x))[:, 0, :]
