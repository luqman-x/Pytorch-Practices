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



def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

def forward(self, idx, target=None):
    B, T = idx.shape
    device = idx.device
    assert T <= self.config.max_len, \
        f"Sequence length {T} exceeds max_len {self.config.max_len}"

     #Token + position embeddings 
    tok = self.transformer.tok_emb(idx)   # [B, T, d_model]
 
    # Position indices: [0, 1, 2, ..., T-1] for every item in batch
    pos_idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    pos = self.transformer.pos_emb(pos_idx)   # [1, T, d_model]
 
    x = self.transformer.drop(tok + pos)   # [B, T, d_model]
 
    #  Causal mask 
    mask = make_causal_mask(T, device)   # [1, 1, T, T]
 
    # N transformer blocks
    for block in self.transformer.blocks:
            x, _ = block(x, mask=mask)   # [B, T, d_model]
 
    # Final LayerNorm 
    x = self.transformer.ln_f(x)   # [B, T, d_model]
 
    # Language model head 
    logits = self.lm_head(x)   # [B, T, vocab_size]
 
    # Loss (training only) 
    loss = None
    if targets is not None:
            # Reshape for CrossEntropyLoss:
            # [B, T, vocab_size] → [B*T, vocab_size]
            # [B, T]             → [B*T]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
 
    return logits, loss
 
def get_shakespeare_data(block_size=128):
    """Download and tokenise tiny Shakespeare dataset."""
    import urllib.request
 
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")
 
    with open("shakespeare.txt", "r") as f:
        text = f.read()
 
    print(f"Dataset: {len(text):,} characters")
 
    # Character-level tokenisation (simple, no tiktoken needed)
    chars    = sorted(set(text))
    vocab_size = len(chars)
    print(f"Vocabulary: {vocab_size} unique characters")
 
    stoi = {c: i for i, c in enumerate(chars)}   # char → int
    itos = {i: c for i, c in enumerate(chars)}   # int → char
 
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
 
    data = torch.tensor(encode(text), dtype=torch.long)
 
    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
 
    return train_data, val_data, vocab_size, encode, decode
 
       

def get_batch(data, block_size, batch_size, device):
    """
    Sample a random batch of (inputs, targets) from data.
    inputs : [B, block_size]  — token indices
    targets: [B, block_size]  — same tokens shifted by 1
 
    For input  [t0, t1, t2, ..., tN-1]
    targets is [t1, t2, t3, ..., tN]
    The model predicts each next token from all previous tokens.
    """
    # Random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i    : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

from tqdm import tqdm

def train_gpt(config: GPTConfig, steps=3000, eval_interval=300,
              batch_size=32, block_size=128):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
 
    # Data
    train_data, val_data, vocab_size, encode, decode = \
        get_shakespeare_data(block_size)
 
    # Override vocab size with actual dataset vocab
    config.vocab_size = vocab_size
    config.max_len    = block_size
 
    model = GPT2(config).to(device)
 
    # AdamW with weight decay (preferred over Adam for transformers)
    # Weight decay regularises all non-bias, non-LayerNorm parameters.
    optimizer = optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.1
    )
 
    # Cosine LR schedule with warmup
    # Warmup: linear increase from 0 → max_lr over warmup_steps
    # Prevents large gradient updates before embeddings have warmed up
    warmup_steps = 100
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps            # linear warmup
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
    train_losses, val_losses = [], []
    best_val = float('inf')
 
    model.train()

    pbar = tqdm(range(steps), desc="Training")

    for step in pbar:
        xb, yb = get_batch(train_data, block_size, batch_size, device)
 
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
 
        # Gradient clipping: caps gradient norm at 1.0
        # Transformers can have explosive gradients — this is standard.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
        optimizer.step()
        scheduler.step()

        # Update tqdm display every iteration
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )
 
        if step % eval_interval == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                # Estimate val loss over multiple batches
                val_loss = sum(
                    model(*get_batch(val_data, block_size, batch_size, device))[1].item()
                    for _ in range(20)
                ) / 20
 
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            lr = optimizer.param_groups[0]['lr']

            pbar.set_postfix(
                train_loss=f"{loss.item():.4f}",
                val_loss=f"{val_loss:.4f}",
                lr=f"{lr:.2e}"
            )
 
            print(f"Step {step:4d} | Train loss: {loss.item():.4f} "
                  f"| Val loss: {val_loss:.4f} | LR: {lr:.2e}")
 
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "gpt2_shakespeare.pth")
 
            model.train()
 
    return model, encode, decode, train_losses, val_losses



@torch.no_grad()
def generate(model, encode, decode, prompt="O Romeo",
             max_new_tokens=200, temperature=0.8, top_k=40):
    """
    Autoregressive text generation:
    Feed prompt → get logits → sample next token → append → repeat
    """
    model.eval()
    device = next(model.parameters()).device
 
    # Encode prompt to token indices
    idx = torch.tensor(encode(prompt), dtype=torch.long,
                       device=device).unsqueeze(0)   # [1, T]
 
    print(f"\n--- Generating from: '{prompt}' ---")
 
    for _ in range(max_new_tokens):
        # Crop to max context window
        idx_cond = idx[:, -model.config.max_len:]
 
        # Forward pass → logits for the last position
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]   # [1, vocab_size] — last token only
 
        # Apply temperature
        logits = logits / temperature
 
        # Top-k: zero out all logits outside the top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
 
        # Sample from the distribution
        probs   = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]
 
        # Append sampled token and continue
        idx = torch.cat([idx, next_tok], dim=1)
 
    generated_text = decode(idx[0].tolist())
    print(generated_text)
    return generated_text

def visualise_attention(model, encode, decode, text="To be or not to be"):
    model.eval()
    device = next(model.parameters()).device
    tokens = encode(text)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    chars = [decode([t]) for t in tokens]
 
    # Forward pass — collect attention weights from ALL blocks
    tok = model.transformer.tok_emb(idx)
    pos_idx = torch.arange(idx.size(1), device=device).unsqueeze(0)
    pos = model.transformer.pos_emb(pos_idx)
    x = tok + pos
    mask = make_causal_mask(idx.size(1), device)
 
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    block_indices = [0, 1, model.config.n_layers//2, model.config.n_layers-1]
 
    for ax_idx, block_i in enumerate(block_indices):
        with torch.no_grad():
            for i, block in enumerate(model.transformer.blocks):
                _, weights = block(x, mask=mask)
                if i == block_i:
                    # Average across heads: [B, n_heads, T, T] → [T, T]
                    attn_map = weights[0].mean(0).cpu().numpy()
                    break
                x, _ = block(x, mask=mask)
 
        ax = axes[ax_idx]
        ax.imshow(attn_map, cmap='Blues', vmin=0, vmax=attn_map.max())
        ax.set_xticks(range(len(chars)))
        ax.set_yticks(range(len(chars)))
        ax.set_xticklabels(chars, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(chars, fontsize=8)
        ax.set_title(f"Layer {block_i} (avg heads)", fontsize=9)
 
    plt.suptitle(f"Attention patterns: '{text}'", fontsize=11)
    plt.tight_layout()
    plt.savefig("attention_maps.png", dpi=150)
    plt.show()
    print("Attention maps saved.")


import os

if __name__ == "__main__":

    
    test_attention()

    checkpoint_path = "gpt2_shakespeare.pth"


    config = GPTConfig(
        d_model  = 128,
        n_layers = 4,
        n_heads  = 4,
        d_ff     = 512,
        dropout  = 0.1,
    )

   
    train_data, val_data, vocab_size, encode, decode = \
        get_shakespeare_data(block_size=128)

    config.vocab_size = vocab_size
    config.max_len = 128

   
    if os.path.exists(checkpoint_path):

        print("\nLoading existing checkpoint...\n")

        model = GPT2(config)

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location="cpu"
            )
        )

        print("Checkpoint loaded successfully.")

    else:

        print("\nNo checkpoint found.")
        print("Starting training from scratch.\n")

        model = None

    
    model, encode, decode, t_losses, v_losses = train_gpt(
        config,
        model=model,
        steps=3000,
        eval_interval=300,
        batch_size=64,
        block_size=128,
    )

    
    plt.figure(figsize=(8, 4))

    plt.plot(t_losses,
             'b-o',
             label='Train loss')

    plt.plot(v_losses,
             'r-o',
             label='Val loss')

    plt.title("GPT-2 Training")
    plt.xlabel("Eval Step")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig("gpt2_training.png", dpi=150)
    plt.show()

    
    generate(
        model,
        encode,
        decode,
        prompt="HAMLET:\n",
        max_new_tokens=300,
        temperature=0.8
    )

   
    visualise_attention(
        model,
        encode,
        decode,
        text="To be or not to be"
    )

   
    print("\n")
    print("=" * 60)
    print("CHAT MODE")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        prompt = input("\nYou: ")

        if prompt.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        response = generate(
            model,
            encode,
            decode,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.8,
            top_k=40
        )

        print("\nModel:")
        print(response)   
 



