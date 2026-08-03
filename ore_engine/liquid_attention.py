"""
Aethelion Core: Liquid Attention Head & Symbolic Gateway
Description: Dynamically allocates ephemeral attention layers without backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SymbolicVerifierGate(nn.Module):
    """
    Motor determinista que evalúa tensores y fuerza máscaras lógicas.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.logic_graph = nn.Parameter(torch.randn(d_model, d_model), requires_grad=False)
        
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        truth_mask = (torch.matmul(attention_scores, self.logic_graph) > 0.0).float()
        truth_mask = truth_mask + 1e-9 
        return torch.log(truth_mask)

class LiquidAttentionHead(nn.Module):
    """
    Capa de atención mutante. Genera sinapsis efímeras en tiempo de inferencia.
    """
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_k = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.symbolic_gate = SymbolicVerifierGate(self.d_k)
        self.register_buffer('ephemeral_memory', torch.zeros(d_model, d_model))

    def forward(self, x: torch.Tensor, requires_crystallization: bool = False) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        
        Q = self.q_proj(x).view(batch_size, seq_length, -1, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, -1, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, -1, self.d_k).transpose(1, 2)
        
        raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        logic_mask = self.symbolic_gate(raw_scores)
        safe_scores = raw_scores + logic_mask
        
        attn = F.softmax(safe_scores, dim=-1)
        output = torch.matmul(attn, V)
        
        if requires_crystallization:
            novelty_delta = torch.matmul(x.transpose(1, 2), output.contiguous().view(batch_size, seq_length, -1))
            self.ephemeral_memory.add_(novelty_delta.mean(dim=0) * 0.01)
            
        final_output = output.contiguous().view(batch_size, seq_length, d_model) + self.ephemeral_memory
        
        return final_output
