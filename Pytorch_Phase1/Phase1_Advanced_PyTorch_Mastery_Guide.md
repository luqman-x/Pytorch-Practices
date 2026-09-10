# Phase 1: Advanced PyTorch Mastery
### Your course guide — taught the way I'd teach a sharp engineer who's never seen this material before

**Where you are:** You've trained CNNs. You built the ICBHI respiratory classifier and the malaria RDT detector with EfficientNet-B0. You know how to call `.backward()` and watch loss go down.

**What this phase does:** It opens the hood. Right now, PyTorch is a set of spells that work. By the end of Phase 1, you'll know *why* they work, what breaks them, and how to write training code the way engineers at labs like FAIR or DeepMind write it — not "code that runs" but "code that is correct, debuggable, and fast."

I'm going to teach this in the order you'll actually need it, with the "why" first, because memorizing API calls without a mental model is how people get stuck for three days on a bug that a 30-second mental-model check would have caught.

---

## 1A — PyTorch Internals: What's Actually Happening

### The single most important idea: the computational graph

When you write:

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()
```

PyTorch isn't just doing arithmetic. Every time a tensor with `requires_grad=True` touches an operation, PyTorch secretly builds a **graph** behind your back — a record of "z came from summing y, y came from multiplying x by 2." This is called **autograd** (automatic differentiation).

Think of it like a receipt printer at a shop. Every operation prints a receipt (a `grad_fn`) that says "here's what I did and here's how to undo it mathematically." When you call `z.backward()`, PyTorch walks backward through that stack of receipts, applying the chain rule at each step, until it reaches `x` and fills in `x.grad`.

**Why this matters practically:** the graph is built *dynamically*, one operation at a time, as your Python code executes (this is PyTorch's famous "define-by-run" design, unlike old TensorFlow's "define-then-run"). This is why you can put `if` statements and loops inside your model's `forward()` and it just works — the graph is whatever code path actually ran.

### `forward()` vs `__call__()` — and why you never touch `__call__`

You write `model(x)`, not `model.forward(x)`, even though you defined `forward()`. Here's why: `nn.Module.__call__()` is the real entry point, and it does bookkeeping around your `forward()` — triggering **hooks** (functions other code can register to run before/after your forward pass, used heavily for things like Grad-CAM, which you already used). If you call `model.forward(x)` directly, you silently skip all of that. Rule: always call `model(x)`, never `model.forward(x)`.

### In-place operations: the silent gradient killer

```python
y = x.relu_()   # the trailing underscore = in-place
```

In-place ops modify a tensor's memory directly instead of creating a new tensor. The problem: autograd's "receipts" sometimes need the *original* values of a tensor to compute a correct gradient later. If you overwrite that tensor in place before backward() runs, PyTorch either throws a `RuntimeError` (if it can detect it) or, worse, silently gives you a wrong gradient. 

**Practical rule:** avoid in-place ops (`+=`, `.relu_()`, `.add_()`) on any tensor that's part of your gradient path, unless you specifically know it's safe (e.g. inside `torch.no_grad()`).

### Leaf vs non-leaf tensors

- A **leaf tensor** is one you created directly (your input data, or `nn.Parameter` weights) — it sits at the "start" of the graph.
- A **non-leaf tensor** is the output of some operation on other tensors (e.g. `y = x * 2` — `y` is non-leaf).

Only leaf tensors accumulate `.grad` by default. If you try to check `.grad` on a non-leaf (like `y` above), it'll be `None` — that's not a bug, that's PyTorch saving memory, because in a huge network you don't want to store gradients for every intermediate activation. If you *need* to inspect a non-leaf's gradient (for debugging), call `y.retain_grad()` before the backward pass.

### Writing a custom autograd Function

99% of the time, PyTorch's built-in ops give you autograd for free. But when you need a custom mathematical operation with a hand-derived gradient (common in medical-imaging loss functions, e.g. a differentiable Dice or IoU with special edge-case handling), you subclass `torch.autograd.Function`:

```python
class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)   # stash what backward() will need
        output = input_tensor ** 2            # example: y = x^2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output * 2 * input_tensor   # dy/dx = 2x, chain rule applied
        return grad_input
```

`ctx` is just a storage box that carries information from `forward()` to `backward()`. You'll use this pattern later (Phase 8) when you write custom CUDA operators.

### Gradient checkpointing: trading compute for memory

Normally, PyTorch stores every intermediate activation during the forward pass so it can reuse them during backward. On a big model (like a 3D U-Net for tumour segmentation), this eats VRAM fast. **Gradient checkpointing** says: "don't store these activations — just re-run the forward pass for this chunk again during backward, when you need it." You trade extra compute time for a lot less memory:

```python
from torch.utils.checkpoint import checkpoint
# instead of: out = self.big_block(x)
out = checkpoint(self.big_block, x, use_reentrant=False)
```

This is exactly how people fit large 3D medical volumes into consumer GPUs.

### `torch.no_grad()` vs `model.eval()` — different jobs, often confused

- `model.eval()` changes the **behavior of specific layers** — BatchNorm stops updating its running statistics, Dropout stops zeroing activations. It does *not* stop gradient tracking.
- `torch.no_grad()` tells autograd to **stop building the graph entirely**, saving memory and compute, but doesn't touch layer behavior.

**Rule you'll use forever:** at inference/validation time, use both together:

```python
model.eval()
with torch.no_grad():
    preds = model(x)
```

Forgetting `model.eval()` during validation is one of the most common silent bugs in medical imaging code — your validation AUC looks unstable because BatchNorm is still using per-batch statistics instead of its learned running averages.

---

## 1B — Writing Production-Grade PyTorch

This section is about the difference between a script that trains a model once, and a training pipeline you can trust, rerun, and hand to someone else.

### Dataset and DataLoader, done properly

You've built `Dataset` classes before. The production-grade version pays attention to:

```python
class RespiratoryDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load one sample. Keep this lightweight — it runs per-item, per-worker.
        spectrogram = load_and_preprocess(self.file_paths[idx])  # (n_mels, T)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        label = self.labels[idx]
        return spectrogram, label

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # parallel CPU processes preparing your next batch
    pin_memory=True,        # speeds up the CPU→GPU transfer
    persistent_workers=True,  # workers survive between epochs — avoids re-spawn overhead
    collate_fn=custom_collate,  # only needed if samples have variable shapes (e.g. variable-length audio)
)
```

- **`num_workers`**: each worker is a separate process loading data in the background *while your GPU is busy training on the previous batch*. Without this, your GPU sits idle waiting for data — a classic and easy-to-miss bottleneck.
- **`pin_memory`**: pins the batch in a special region of CPU RAM that the GPU can read faster.
- **`collate_fn`**: DataLoader's default assumes every item in a batch is the same shape. If you're padding variable-length audio clips or handling variable numbers of detected regions, you write a custom `collate_fn` to stack them correctly.

### Custom loss functions

For imbalanced medical data (way more "healthy" than "disease" samples), plain cross-entropy is a poor choice — it lets the model get away with just predicting the majority class. You already know this from the malaria project. Here's the standard toolkit:

```python
# Focal loss — down-weights easy/confident examples, forces the model to focus on hard ones
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)                       # model's confidence in the correct answer
        focal_term = (1 - p_t) ** self.gamma         # shrinks loss for already-confident predictions
        loss = self.alpha * focal_term * bce
        return loss.mean()

# Dice loss — directly optimizes overlap, standard for segmentation masks
def dice_loss(pred_probs, target_mask, eps=1e-6):
    intersection = (pred_probs * target_mask).sum()
    union = pred_probs.sum() + target_mask.sum()
    return 1 - (2 * intersection + eps) / (union + eps)

# Combined loss — BCE gives stable gradients early, Dice pushes overlap quality
def combined_loss(logits, target_mask):
    bce = F.binary_cross_entropy_with_logits(logits, target_mask)
    dice = dice_loss(torch.sigmoid(logits), target_mask)
    return bce + dice
```

### Learning rate schedulers: which one, when

- **`CosineAnnealingWarmRestarts`**: LR smoothly decreases like a cosine curve, then periodically "restarts" to a high value. Good for longer training runs where you want the model to periodically escape shallow local minima.
- **`OneCycleLR`**: LR ramps *up* first, then *down*, in one single cycle across your whole training run. This is often the fastest way to converge for a fixed training budget — it's become close to a default choice in modern training recipes.

Rule of thumb: if you know your total training steps in advance and want the best result fastest → `OneCycleLR`. If you're doing long exploratory training and want periodic "resets" → `CosineAnnealingWarmRestarts`.

### Gradient clipping

Occasionally a batch produces a huge gradient (numerical instability, a bad sample, an unstable loss landscape early in training) and it can blow up your weights in one step. Clipping caps the gradient's overall size before the optimizer applies it:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Cheap insurance, always worth having, especially for anything with recurrent layers or transformers.

### Mixed precision training — the full picture

Your GPU can do math faster in 16-bit floating point (FP16/BF16) than 32-bit, and it uses half the memory. The risk: some operations (like summing many small gradients) lose precision in FP16 and become unstable. **Automatic Mixed Precision (AMP)** solves this by keeping the *risky* operations in FP32 and running everything else in FP16, automatically:

```python
scaler = torch.cuda.amp.GradScaler()

for batch, labels in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():          # this block runs in mixed precision
        outputs = model(batch)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()            # scales the loss up before backward (prevents tiny FP16 gradients from vanishing to 0)
    scaler.step(optimizer)                    # unscales, then steps
    scaler.update()                           # adjusts the scale factor for next time
```

The `GradScaler` exists because in FP16, very small gradient values can literally round down to zero. Scaling the loss up before `.backward()` keeps those small numbers representable, then it's undone before the optimizer actually updates weights.

### Gradient accumulation — simulating a bigger GPU

If your model needs a batch size of 64 for stable training but your GPU only fits 16 at a time, you can accumulate gradients across 4 mini-batches before stepping:

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (batch, labels) in enumerate(loader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps   # scale down so the sum matches a real batch-of-64 loss
    loss.backward()                                          # gradients accumulate in .grad by default
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Checkpointing that actually lets you resume training

A checkpoint that only saves `model.state_dict()` lets you *use* the model but not properly *resume training* — you'd lose the optimizer's momentum state and the scheduler's position. Save all of it:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_auc': best_val_auc,
}, f'checkpoint_epoch_{epoch}.pt')

# to resume:
ckpt = torch.load('checkpoint_epoch_10.pt')
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch'] + 1
```

---

## 1C — Profiling and Debugging

### `torch.profiler` — finding out where time actually goes

Don't guess where your training loop is slow. Measure it:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for batch, labels in loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

This tells you, operation by operation, how much time is CPU-side (often data loading) vs GPU-side (actual compute). If data loading dominates, more `num_workers` helps. If compute dominates, mixed precision or a smaller model helps. Never optimize blind.

### Experiment tracking: wandb and TensorBoard

You already use Weights & Biases — keep going deeper. Log not just loss and accuracy, but also learning rate, gradient norms, and example predictions each epoch. The goal is that six months from now you can look at a run and understand exactly what happened without re-running it.

### Memory profiling and leaks

```python
print(torch.cuda.memory_summary())
```

A classic memory leak pattern: accumulating loss values across a whole epoch by doing `total_loss += loss` where `loss` is still attached to the computational graph. This keeps every batch's entire graph alive in memory. Fix: `total_loss += loss.item()` — `.item()` pulls out a plain Python float, detached from the graph.

### Debugging NaN losses

When loss becomes `NaN`, don't guess — instrument:

```python
torch.autograd.set_detect_anomaly(True)   # slows training, but tells you EXACTLY which op produced the NaN
```

Turn this on temporarily when you hit a NaN, find the offending operation from the stack trace it prints, then turn it off again (it has real performance cost, so it's a debugging tool, not something to leave on).

Common NaN causes in your kind of work: `log(0)` in a loss function (add a small epsilon), a learning rate that's too high, or unnormalized input data with extreme outlier values.

---

## 1D — The Elite Code Pattern: Your Standard Training Script

This is the shape every training script you write from now on should follow. It's not decoration — the numbered phases make a script skimmable in 10 seconds, and the `CONFIG` dict means every hyperparameter lives in exactly one place instead of scattered as magic numbers through your code.

```python
CONFIG = {
    'seed': 42,
    'batch_size': 32,
    'lr': 3e-4,
    'epochs': 50,
    'num_workers': 4,
    'mixed_precision': True,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(CONFIG['seed'])   # reproducibility — always, no exceptions

# === Phase 1: Data ===
# === Phase 2: Model ===
# === Phase 3: Optimizer + Scheduler + Scaler ===
# === Phase 4: Training Loop ===
# === Phase 5: Validation Loop ===
# === Phase 6: Logging + Checkpointing ===
# === Phase 7: Test + Clinical Metrics ===
```

Inside each phase, follow the two habits that catch bugs before they cost you a day of confusion:
1. **Shape comments on every tensor transformation** — `# (B, 1, T) -> (B, n_mels, T // hop)`
2. **Assert statements before critical operations** — `assert x.shape == (B, C, H, W), f'Expected (B,C,H,W), got {x.shape}'`

---

## Mini-Projects for Phase 1

These map directly onto work you've already built, so you're not starting from zero — you're upgrading what exists.

### Project 1 — Upgrade the ICBHI respiratory classifier
Take your existing training script and rebuild it with: mixed precision (`autocast` + `GradScaler`), gradient accumulation, full checkpoint saving (model + optimizer + scheduler state), and shape-annotated tensors throughout.
**Then profile it** with `torch.profiler` and write down, in your README, exactly where the time goes — data loading vs forward vs backward.

### Project 2 — Implement focal loss and combined BCE+Dice loss from scratch
Apply it to the malaria RDT project. To sanity-check that your custom gradient math is right, use **finite-difference gradient checking**: numerically perturb an input by a tiny epsilon, see how much the loss changes, and confirm it roughly matches what `.backward()` computed analytically. This is the standard way to catch a wrong hand-derived gradient before it silently corrupts a week of training.

### Project 3 — Write a custom DataLoader for a medical imaging dataset
Requirements: handles class imbalance (via `WeightedRandomSampler`, which you've used before, or a custom sampler), applies on-the-fly augmentation in `__getitem__`, and caches preprocessed data to disk (e.g. converting raw images to preprocessed tensors once, saved as `.pt` files, so you don't redo expensive preprocessing every epoch).

---

## Resources for This Phase

- PyTorch official docs: *Autograd Mechanics* and *Extending PyTorch* — read both in full, they're short and directly answer "why" questions this guide only summarizes.
- Christian Perone's "PyTorch Under the Hood" blog series (christianperone.com) — internals, explained by someone who's read the source.
- Andrej Karpathy, "A Recipe for Training Neural Networks" — a short, famous essay on debugging discipline. Worth memorizing, not just reading once.
- fast.ai course, Part 2 (from-scratch implementations) — reinforces everything here by having you rebuild it yourself.

---

## Where to start today

1. Open your ICBHI classifier. Rewrite the training loop with the full `CONFIG` dict pattern.
2. Add shape-annotation comments on every tensor operation.
3. Add `assert` statements before every major transformation.
4. Add mixed precision training with `torch.cuda.amp`.
5. Profile it with `torch.profiler` and find out where the time actually goes.

That's it — that's Day 1. When you've done this and you're ready, tell me and we'll move into Phase 2: Deep Learning Architecture Engineering (ResNet and U-Net from scratch, uncertainty calibration, and explainability — all directly relevant to the clinical work you're already doing).
