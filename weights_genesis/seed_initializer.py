"""
Aethelion Weights Genesis: Seed Initializer
Description: Initializes the baseline structural tensor weights for Aethelion.
"""

import torch

def initialize_seed_weights(d_model: int) -> torch.Tensor:
    """
    Initializes orthogonal baseline weights to establish the structural genesis.
    """
    base_tensor = torch.empty(d_model, d_model)
    torch.nn.init.orthogonal_(base_tensor)
    return base_tensor
