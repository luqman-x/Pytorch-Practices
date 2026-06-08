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

class TransformerBlock(nn.Module):
    """
    One GPT-2 transformer block:
        x → LayerNorm → MultiHeadAttention → + residual
          → LayerNorm → FeedForward         → + residual
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        attn_out, weights = self.attn(
            self.ln1(x),
            mask=mask
        )
        x = x + attn_out

        x = x + self.ff(self.ln2(x))

        return x, weights
    
# SINUSOIDAL POSITIONAL ENCODING

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(
            0, max_len
        ).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(1000.0) / d_model)
                        )
        
        pe[:, 0::2] = torch.sin(pos * div)   # even dims → sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd dims  → cos


        
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]
 
    def forward(self, x):
        # x: [B, T, d_model]
        # pe[:, :T]: take only the first T position vectors
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
# ROPE: ROTARY POSITIONAL ENCODING
def precompute_rope_frequencies(d_k: int, max_len: int,
                                 base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))

    positions = positions = torch.arange(max_len).float()

    freqs = torch.outer(positions, theta)

    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()   # both [max_len, d_k]




def apply_rope(x, cos, sin):
    """
    Apply rotary position embedding to Q or K.
    x   : [B, n_heads, T, d_k]
    cos, sin: [T, d_k]
    """
    # Split x into two halves — we rotate pairs of dimensions
    d_k = x.size(-1)
    x1  = x[..., :d_k // 2]   # first half
    x2  = x[..., d_k // 2:]   # second half
 
    # Rotate: [x1, x2] → [-x2, x1] (90° rotation in each 2D plane)
    x_rotated = torch.cat([-x2, x1], dim=-1)
 
    # Broadcast cos/sin from [T, d_k] to [B, n_heads, T, d_k]
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
 
    return x * cos + x_rotated * sin


# FULL GPT-2 MODEL
@dataclass
class GPTConfig:
    vocab_size  : int   = 50257    # GPT-2 vocabulary size
    max_len     : int   = 1024     # maximum sequence length
    d_model     : int   = 768      # embedding dimension
    n_layers    : int   = 12       # number of transformer blocks
    n_heads     : int   = 12       # number of attention heads
    d_ff        : int   = 3072     # FFN hidden dimension (4 × d_model)
    dropout     : float = 0.1
    # Tiny config for training on CPU/small GPU:
    # d_model=128, n_layers=4, n_heads=4, d_ff=512


class GPT2(nn.model):
    def __int__(self,  config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb = nn.Embedding(config.vocab_size, config.d_model),
                pos_emb = nn.Embedding(config.max_len, config.d_model),
                drop    = nn.Dropout(config.dropout),

                # N transformer blocks
            blocks  = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]),
            ln_f    = nn.LayerNorm(config.d_model),


        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.transformer.tok_emb.weight

        self.apply(self._init_weights)

        total = sum(p.numel() for p in self.parameters())
        print(f"GPT-2 initialised | Parameters: {total:,}")




       












