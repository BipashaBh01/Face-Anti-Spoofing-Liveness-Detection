# Import all necessary packages
from lib_imports import *
# from custom_module_imports import *
from Functions.config import *


#=========================================================
# MAIN CLASS FOR STREAM GATE
#=========================================================

import torch
import torch.nn as nn

# -- Cross-stream attention gating ---------------------------------------
class StreamGating(nn.Module):
    """
    StreamGating module for adaptive fusion of multiple feature streams.

    This module learns how much importance to give each input stream
    (e.g., RGB, Optical Flow, Frequency features) at every frame and for
    every sample in the batch.

    Instead of treating all streams equally, it dynamically assigns
    weights using a learned gating mechanism.

    Input:
        streams (list of torch.Tensor):
            A list of tensors, one per stream.
            Each tensor has shape: (B, T, D)
            where:
                B = batch size
                T = number of frames (temporal length)
                D = feature dimension

    Output:
        torch.Tensor:
            Fused representation of shape (B, T, D)
            computed as weighted sum of all streams.
    """

    def __init__(self, n_streams=CFG.N_STREAMS, dim=CFG.PROJ_DIM):
        """
        Args:
            n_streams (int): Number of input streams to fuse (We expect 3: RGB, FFT, Flow).
            dim (int): Feature dimension of each stream (D).
        """
        super().__init__()   # Initialises parent class (`nn.Module`). Without this, PyTorch won’t track parameters properly.

        # Gate network:
        self.gate = nn.Sequential(

            # Linear layer: 
            # Input size = n_streams * dim. (3 streams × 256 dim = 768 input features)
            # Output size = n_streams = 3. (So it produces: [weight_RGB, weight_Flow, weight_FFT] for every frame)
            # Softmax Layer: Converts raw scores into probabilities 
            # Example: [0.7, 0.2, 0.1] Meaning: RGB is most important, Flow is less important, FFT is least important
            nn.Linear(n_streams * dim, n_streams),
            nn.Softmax(dim=-1)
        )

    def forward(self, streams):
        """
        Forward pass for stream fusion.

        Args:
            streams (list of torch.Tensor):
                List of tensors [(B, T, D), ...] of length n_streams

        Returns:
            torch.Tensor:
                Fused tensor of shape (B, T, D)
        """

        # Combine all streams into one tensor (a new dimension/stream dimension)
        # Three tensors each having (B,T,D). Result shape: (B, T, n_streams, D) or (B,T,3,D)
        # Now all streams are inside one tensor.
        stacked = torch.stack(streams, dim=2)

        # Concatenate features from all streams along feature dimension (Join features side by side)
        # Before: (B, T, D) + (B, T, D) + (B, T, D); After: (B, T, 3D)
        # Shape: (B, T, n_streams * D)
        # This is done to give all information to the gating network. It uses this to decide importance of each stream.
        concat = torch.cat(streams, dim=-1)

        # Compute attention-like weights for each stream
        # Pass through gate:- Input: (B, T, 3D), Output: (B, T, 3). Or each frame, model predicts: [weight1, weight2, weight3]
        # Add extra dimension .unsqueeze(-1):- Before: (B, T, 3), After: (B, T, 3, 1). This is done to match shape with stacked, so that multiplication works (broadcasting).
        # Output shape: (B, T, n_streams)
        weights = self.gate(concat).unsqueeze(-1)  # (B, T, n_streams, 1)

        # Each stream is multiplied by its weight (Broadcasting over feature dimension D)
        # Broadcasting automatically applies weight across D features
        # Multiply: stacked -> (B, T, 3, D), weights -> (B, T, 3, 1).
        # .sum(dim=2): Add all streams. Removes the stream dimension. Before: (B, T, 3, D), After: (B, T, D). Now all streams are combined into one
        fused = (stacked * weights).sum(dim=2)  # Output: (B, T, D)

        return fused    # Returns final fused output