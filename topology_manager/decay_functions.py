"""
Aethelion Topology Manager: Decay Functions
Description: Manages programmed forgetting for obsolete graph data.
"""

import torch

def exponential_decay(ephemeral_memory: torch.Tensor, decay_factor: float = 0.99):
    """
    Applies a decay factor to ephemeral synapses to prevent unbounded memory growth.
    """
    return ephemeral_memory * decay_factor
