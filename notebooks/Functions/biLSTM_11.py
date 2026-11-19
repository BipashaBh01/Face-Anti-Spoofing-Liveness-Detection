# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *

#=========================================================
# MAIN CLASS FOR BiLSTM
#=========================================================

class BiLSTMEncoder(nn.Module):
    """
    BiLSTM Encoder with Attention Mechanism.

    This module:
    1. Processes sequential input using a Bidirectional LSTM
    2. Applies an attention mechanism to focus on important timesteps
    3. Produces a fixed-size context vector (sequence summary)

    Args:
        input_dim (int): Dimension of input features (D)
        hidden_dim (int): Hidden size of LSTM
        num_layers (int): Number of stacked LSTM layers

    Input:
        x (torch.Tensor): Shape (B, T, D)
            B = batch size
            T = sequence length (timesteps)
            D = feature dimension

    Output:
        torch.Tensor: Shape (B, 2 * hidden_dim)
            Final context vector after attention pooling
    """

    def __init__(self, input_dim=CFG.PROJ_DIM, hidden_dim=CFG.RNN_HIDDEN, num_layers=CFG.RNN_LAYERS):
        # Initialises the model
        # input_dim: feature size of each timestep
        # hidden_dim: LSTM hidden size
        # num_layers: number of stacked LSTM layers
        
        super().__init__()  # Calls parent class constructor. Required in every PyTorch model.

        # -------------------- BiLSTM Layer --------------------
        # Bidirectional=True --> captures both past and future context
        # Output shape: (B, T, 2 * hidden_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,                             # input shape is (B, T, D) instead of (T, B, D)
            bidirectional=True,                           # Processes sequence forward + backward. Output becomes 2 * hidden_dim
            dropout=CFG.DROPOUT if num_layers > 1 else 0  # Dropout only applied if multiple layers
        )

        # -------------------- Attention Network --------------------
        # Computes importance (attention score) for each timestep
        # Input: (B, T, 2 * hidden_dim)
        # Output: (B, T, 1) -> attention weights before softmax
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),  # Reduce dimension: Input: (B, T, 2H) --> Output: (B, T, H/2)
            nn.Tanh(),                                  # Non-linearity
            nn.Linear(hidden_dim // 2, 1, bias=False)   # Produces 1 score per timestep. Output: (B,T,1). This score tells how important each timestep is.
        )

        # -------------------- Dropout Layer --------------------
        self.drop = nn.Dropout(CFG.DROPOUT)                  # Randomly drops neurons during training. (Prevents Overfitting)

        # Output feature dimension (because BiLSTM doubles hidden size, Final feature size = 2H)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        """
        Forward pass.

        Steps:
        1. Pass input through BiLSTM -> get sequence outputs
        2. Compute attention scores for each timestep
        3. Convert scores to probabilities using softmax
        4. Compute weighted sum of sequence outputs
        5. Apply dropout

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D)

        Returns:
            torch.Tensor: Context vector of shape (B, 2 * hidden_dim)
        """

        # Step 1: Pass input through LSTM
        # Input: (B, T, D), 
        # Output: out: (B, T, 2 * hidden_dim), _ : final hidden state (not used here)
        out, _ = self.lstm(x)

        # Step 2: Compute attention scores, shape -> (B, T, 1)
        # softmax(..., dim=1): Converts scores into probabilities across time. Ensures: All weights sum to 1. Important timesteps get higher weight
        w = F.softmax(self.attn(out), dim=1)

        # Step 3: Apply attention weights
        # Element-wise multiplication and sum over time
        # (out * w) -> (B, T, 2 * hidden_dim)
        # sum(dim=1) -> (B, 2 * hidden_dim)
        context = (out * w).sum(dim=1)

        # Step 4: Apply dropout and return
        return self.drop(context)