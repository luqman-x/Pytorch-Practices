import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = weights @ V
        return output, weights
    
    
def test_attention():
    B, T, d_k = 2, 5, 64
    H=8
    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_k)
    out, w = scaled_dot_product_attention(Q, K, V)

    print(f"Attention output: {out.shape}")   # [2, 8, 5, 64]
    print(f"Weights sum (should be 1.0): {w[0,0,0].sum():.4f}")



def make_causual_mas(T: int, device: torch.device) -> torch.Tensor:
    """
    Returns an upper-triangular boolean mask of shape [1, 1, T, T].
    True = this position is masked (forbidden to attend to).
 
    For T=4, the mask looks like:
        [[False  True  True  True],
         [False False  True  True],
         [False False False  True],
         [False False False False]]
 
    Token 0 sees only itself.
    Token 3 sees tokens 0, 1, 2, 3.
    """

    mask = torch.triu(torch.ones(T,T, dtype=torch.bool, device=device),
                      diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads: int, dropout: float =0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_input = None, mask= None):
        B, T, _ = x.shape
        kv = kv_input if kv_input is not None else x
        s = kv.size(1)

        Q = self.W_q(x)
        K = self.W_k(kv)
        V = self.W_v(kv)

        #Reshape into heads
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        output = self.W_o(attn_out)   # [B, T, d_model]
        output = self.dropout(output)
 
        return output, weights 
    

# FEED-FORWARD BLOCK (MLP)
# After attention, every position goes through an identical MLP.
# This is where the model stores "facts" — attention routes information,
# the MLP processes and stores it.
#
# Size: 4 × d_model (GPT-2: 768 → 3072 → 768)
# Activation: GELU (smoother ReLU — used in all modern transformers)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),

        )

    def forward(self, x):
        return self.net(x)
    

# TRANSFORMER BLOCK














