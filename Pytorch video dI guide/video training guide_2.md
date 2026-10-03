# Training Video Neural Networks in PyTorch — A Practitioner's Guide (2026)

*Worked examples use a clinical framing (surgical phase recognition from laparoscopic video) since that's a natural fit given your biomedical background — the same code applies to action recognition, sports analytics, or any other video task with minimal changes.*

---

## Table of Contents
1. Video Data Fundamentals
2. Dataset Organization & Preparation
3. Video Loading in PyTorch
4. Preprocessing
5. Architectures for Video
6. Training Pipeline
7. Efficiency & Scalability
8. Advanced Techniques
9. Evaluation
10. Complete Practical Project (code)
11. Best Practices vs. Outdated Techniques
12. Hardware-Dependent Pipeline Changes
13. Learning Roadmap

---

## 1. Video Data Fundamentals

**Representation.** A video is a sequence of frames, each an `[H, W, C]` array, at some **frame rate (FPS)**. What matters for deep learning is not the file on disk but the *decoded tensor* you eventually feed the model: `[T, C, H, W]` per clip, batched to `[B, T, C, H, W]`.

- **Containers vs. codecs** — a container (`.mp4`, `.mkv`, `.avi`) is a box; the codec (`H.264`, `H.265/HEVC`, `AV1`, `VP9`) is how the pixels are compressed inside it. This distinction matters practically: H.264 has near-universal hardware decode support (including on cloud GPUs via NVDEC); AV1 is more efficient on disk but decode is often CPU-bound unless you have recent hardware/driver support. For a training pipeline you re-decode millions of times, so codec choice is a real bottleneck decision, not a formality.
- **Frames are not independent** — unlike images, consecutive frames are highly redundant (temporal correlation). This is *why* video models exist as a distinct field: naively running a 2D CNN per frame throws away motion information, and naively storing every frame as a raw image wastes enormous disk space versus keeping the compressed video and decoding on demand.
- **Key vs. inter frames (I/P/B-frames)** — compressed video stores full "key frames" (I-frames) periodically and encodes other frames as diffs. Seeking to an arbitrary frame requires decoding forward from the nearest preceding I-frame. This is *the* reason random single-frame access in long videos is slow, and why clip-based (contiguous) sampling is standard practice rather than an implementation detail.

**Label granularity** — the task determines everything downstream:

| Task | Label attaches to | Example |
|---|---|---|
| Video classification | Whole clip/video | "This is a cholecystectomy" |
| Action/phase recognition | Segment (start–end time) | "Dissection phase: 04:12–09:47" |
| Temporal event detection | Point or short interval | "Clip application at 07:03" |
| Object tracking | Per-frame bounding box + track ID | Instrument tip location per frame |
| Dense/frame-level classification | Every frame | Frame-by-frame surgical phase |

Action recognition and video classification are architecturally similar (both reduce a clip to a label); temporal detection and tracking require frame-level or region-level outputs and usually a different head (or a fully different model family, e.g., detection-by-tracking). Don't assume one architecture serves all four — the loss function and evaluation protocol diverge sharply.

**Annotation formats.** For classification: a CSV/JSON mapping `video_id → label`. For temporal tasks: interval lists (`start_frame, end_frame, label`), commonly stored as COCO-style JSON extended with temporal fields, or dataset-specific formats (ActivityNet, Kinetics use different schemas — there's no single standard, so plan to write a thin adapter layer around whatever format your annotators/vendor produce).

---

## 2. Dataset Organization & Preparation

Recommended on-disk layout — keep raw video separate from any derived artifacts (indices, cached frames), so you can regenerate the latter without re-touching source data:

```
dataset_root/
├── videos/
│   ├── case_0001.mp4
│   ├── case_0002.mp4
│   └── ...
├── annotations/
│   ├── train.csv        # video_id, start_frame, end_frame, label
│   ├── val.csv
│   └── test.csv
├── metadata.json         # per-video: fps, duration, resolution, patient/session id, split-relevant grouping key
└── splits/
    └── split_seed42.json
```

**Preventing leakage — the single most common bug in video ML.** Two failure modes are specific to video and easy to miss:

1. **Clip-level leakage**: splitting *clips* rather than *source videos* into train/val puts adjacent, near-duplicate clips from the same video on both sides of the split. The model then "memorizes" the source video rather than learning the task, and validation accuracy is inflated and meaningless. **Split at the video (or patient/session) level, never at the clip level.**
2. **Group leakage**: in clinical data specifically, multiple videos from the same patient, same surgeon, or same OR/camera setup can share confounding signal (lighting rig, camera brand, patient anatomy) that lets a model "cheat" by learning the confound instead of the task. Use `GroupKFold`-style splitting keyed on patient/session ID, not just video ID, whenever such groupings exist.

```python
# splits/make_splits.py
"""Group-aware splitting to prevent leakage across patients/sessions."""
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

def make_group_split(metadata_csv: str, group_col: str = "patient_id", seed: int = 42):
    df = pd.read_csv(metadata_csv)  # columns: video_id, patient_id, label, ...
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df[group_col]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    # split temp into val/test, still grouped
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    assert set(train_df[group_col]) & set(val_df[group_col]) == set()   # sanity check: no shared groups
    assert set(train_df[group_col]) & set(test_df[group_col]) == set()
    return train_df, val_df, test_df
```

**Class imbalance** (e.g., a rare surgical complication phase vs. routine dissection) is *worse* in video than images because clip sampling can further dilute rare events. Options, roughly in order of how often they're used in practice:
- **Weighted sampling** (`WeightedRandomSampler`) so rare-class clips are seen more often per epoch — cheap, usually the first thing to try.
- **Class-weighted loss** (weight in `CrossEntropyLoss`) — combine with sampling rather than as a substitute; they address slightly different parts of the problem (batch composition vs. gradient magnitude).
- **Focal loss** — down-weights easy majority-class examples; useful when imbalance is extreme (>50:1) and weighted sampling alone still under-trains the rare class.
- **Oversampling rare *events* at the clip-extraction stage** (sample more overlapping clips centered on rare-label timestamps) — this is video-specific and often more effective than reweighting after the fact, because it fixes the actual scarcity of *distinct* rare examples, not just their sampling frequency.

**Corrupted/missing videos** — always validate the dataset before training, not during (a crash 6 hours into an epoch is expensive):

```python
import decord
from pathlib import Path

def audit_videos(video_dir: str) -> list[str]:
    """Returns list of unreadable video paths."""
    bad = []
    for path in Path(video_dir).glob("*.mp4"):
        try:
            vr = decord.VideoReader(str(path))
            _ = len(vr)              # forces header parse
            _ = vr[0]                # forces first-frame decode
        except Exception as e:
            bad.append(str(path))
    return bad
```

**Reproducibility.** Pin: the split file (with seed), the exact annotation file hash, library versions (`decord`, `torch`, `torchvision`), and the sampling strategy (clip length, stride). A model trained on "the dataset" without pinning the clip-sampling scheme is not actually reproducible — two runs with different random clip offsets per video can produce meaningfully different results, especially with small datasets.

---

## 3. Video Loading in PyTorch

**Decoder comparison** (this changes yearly — verify current benchmarks before committing, but as of 2026 the practical hierarchy is):

| Tool | Speed | GPU decode | Ease of use | When to use |
|---|---|---|---|---|
| OpenCV (`cv2.VideoCapture`) | Slow–moderate | No (CPU only, in practice) | Simple | Prototyping, small datasets, when you need frame-exact seeking simplicity |
| torchvision `io.VideoReader` (`video_reader` backend, PyAV under the hood by default) | Moderate | Depends on backend build | Native PyTorch integration | Small-to-medium projects already in the torchvision ecosystem |
| **Decord** | Fast, especially with `num_threads` and batched frame access | Optional NVDEC via `decord.gpu(0)` | Moderate | Default recommendation for most training pipelines in 2026 — good random-access performance for clip sampling |
| PyAV | Fast, low-level FFmpeg bindings | Manual | More code, more control | When you need custom container/codec handling Decord doesn't expose |
| NVIDIA DALI | Fastest for large-scale training | Yes, full GPU pipeline (decode + resize + augment) | Steeper setup | Large-scale/multi-GPU training where CPU decode is the bottleneck |

**Why this matters concretely**: with a 3D CNN or video transformer, GPU compute per batch is often *fast* relative to CPU frame decoding + augmentation. If your `DataLoader` can't keep up, you get low GPU utilization no matter how good the model or how big the GPU — this is the single most common efficiency failure in video training, more common than it is in image training, precisely because decoding is heavier than reading a JPEG.

**Custom `Dataset` with clip sampling:**

```python
# datasets/surgical_video_dataset.py
"""
Video Dataset with configurable temporal sampling.

Tensor shapes annotated inline:
  clip:  [T, H, W, C]  uint8   -> after transform: [C, T, H, W]  float32
"""
import decord
import numpy as np
import pandas as pd
import torch
from decord import VideoReader
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")  # decord returns torch tensors directly, skips numpy round-trip


class SurgicalPhaseClipDataset(Dataset):
    def __init__(
        self,
        annotations_csv: str,
        video_dir: str,
        num_frames: int = 16,
        sampling: str = "uniform",     # "uniform" | "random" | "dense"
        frame_stride: int = 2,          # used by "dense" sampling
        transform=None,
        train: bool = True,
    ):
        self.df = pd.read_csv(annotations_csv)   # video_id, start_frame, end_frame, label
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.sampling = sampling
        self.frame_stride = frame_stride
        self.transform = transform
        self.train = train
        self.label_to_idx = {l: i for i, l in enumerate(sorted(self.df["label"].unique()))}

    def __len__(self) -> int:
        return len(self.df)

    def _sample_indices(self, start: int, end: int) -> np.ndarray:
        """Returns frame indices to decode, shape [num_frames]."""
        span = max(end - start, self.num_frames)
        if self.sampling == "uniform":
            # deterministic, evenly spaced — standard for val/test
            return np.linspace(start, start + span - 1, self.num_frames).astype(int)
        elif self.sampling == "random":
            # random contiguous-ish offset within each of num_frames bins — standard for train
            bin_size = span / self.num_frames
            offsets = np.random.uniform(0, bin_size, self.num_frames) if self.train else np.zeros(self.num_frames)
            return (start + np.arange(self.num_frames) * bin_size + offsets).astype(int)
        elif self.sampling == "dense":
            # contiguous clip at fixed stride — needed when local motion (not global context) matters
            return np.arange(start, start + self.num_frames * self.frame_stride, self.frame_stride)
        raise ValueError(f"Unknown sampling: {self.sampling}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vr = VideoReader(f"{self.video_dir}/{row.video_id}.mp4", num_threads=1)  # num_threads=1: DataLoader workers parallelize instead
        indices = np.clip(self._sample_indices(row.start_frame, row.end_frame), 0, len(vr) - 1)

        clip = vr.get_batch(indices)              # [T, H, W, C] uint8, torch tensor (decord bridge)
        clip = clip.permute(3, 0, 1, 2).float()    # -> [C, T, H, W], matches torchvision video conventions

        if self.transform:
            clip = self.transform(clip)

        label = self.label_to_idx[row.label]
        return clip, label
```

**Why `num_threads=1` inside the dataset**: parallelism should come from `DataLoader(num_workers=N)` (process-level), not from decord's internal threading fighting for the same CPU cores — oversubscribing both simultaneously is a common and easy-to-miss cause of *worse* throughput than either alone.

---

## 4. Preprocessing

**Standard tensor layout**: `[B, T, C, H, W]` for most video model implementations (matches how `nn.Conv3d` expects `[B, C, T, H, W]` after one more permute — libraries differ on `T` vs `C` ordering, so always check the exact convention the specific model implementation expects before writing your pipeline; this is a frequent source of silent shape-mismatch bugs that don't error, they just train a nonsense model).

**Augmentation — spatial vs. temporal:**
- *Spatial* (per-frame, but applied **consistently across all frames in a clip** — this is the key difference from image augmentation): random crop, flip, color jitter. Apply the *same* random crop/flip to every frame in the clip; independent per-frame augmentation destroys the temporal coherence the model is supposed to learn from.
- *Temporal*: random clip start offset (already in the sampler above), temporal jitter (small per-frame index perturbation), frame dropout, speed perturbation (sample every 2nd frame as if it were the whole clip, simulating faster/slower motion), temporal reversal (task-dependent — reversing a "grasping" action changes its label, so this augmentation is not universally safe, unlike a horizontal flip).

```python
# transforms/video_transforms.py
"""Clip-consistent spatial transforms — the same random params apply to every frame."""
import torch
import torchvision.transforms.v2.functional as F


class ClipRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [C, T, H, W]
        _, _, h, w = clip.shape
        top = torch.randint(0, max(h - self.size, 1), (1,)).item()
        left = torch.randint(0, max(w - self.size, 1), (1,)).item()
        return clip[:, :, top:top + self.size, left:left + self.size]   # same crop window for all T frames


class ClipRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return clip.flip(dims=[-1])    # flip W dim, applied identically to all T frames
        return clip


class ClipNormalize:
    def __init__(self, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip / 255.0 - self.mean) / self.std
```

**Variable-length videos**: don't pad to a fixed number of frames by default — that wastes compute on padding tokens and, for attention-based models, requires masking or the model will attend over meaningless padded frames. Prefer fixed-length *clip sampling* (as above) over padding whenever the task allows it (i.e., whenever the label is clip-level, not full-video-level). Padding is mainly needed for architectures that must consume a whole variable-length video at once (e.g., full-video temporal detection); in that case pad with masking, not with meaningless augmentation.

**Memory-efficient pipelines**: decode at a lower resolution than your final training resolution *if the source supports it* is a myth worth debunking — hardware decoders decode at the encoded resolution regardless; the savings come from resizing immediately after decode, before any further per-frame work, and from decoding fewer frames per clip rather than lowering resolution. If disk space and re-encoding cost are acceptable, pre-resizing your entire dataset once (offline) to your target training resolution removes this cost from every future epoch — a worthwhile one-time investment for datasets you'll train on repeatedly.

---

## 5. Architectures for Video

| Family | How it works | Compute cost | When to use |
|---|---|---|---|
| **CNN + temporal pooling/LSTM** (e.g., ResNet-2D per frame → LSTM or temporal pooling) | 2D CNN extracts per-frame features, a temporal model aggregates them | Low–moderate | Small datasets, limited compute, when motion is not the primary signal (e.g., "is a tool present" vs. "what motion is happening") |
| **3D CNNs** (C3D, I3D, SlowFast) | Convolutions operate jointly over space *and* time (`Conv3d`) | High (3D kernels are expensive) | Motion/dynamics matter a lot; medium-to-large datasets; SlowFast in particular balances a fast, lightweight temporal-resolution pathway with a slow, semantically rich pathway — a good default when compute is constrained but temporal detail matters |
| **ConvLSTM / recurrent** | Convolutional features fed through a recurrent unit over time | Moderate, but sequential (hard to parallelize over T) | Streaming/online inference where you must produce output *during* the video, not just after seeing the whole clip |
| **Two-stream (RGB + optical flow)** | Separate CNNs for appearance (RGB) and motion (optical flow), fused late | High (flow computation is itself expensive) | Historically strong for action recognition; largely superseded by 3D CNNs and transformers that learn motion implicitly, but still used when explicit motion cues are unusually important and flow can be precomputed offline |
| **Video Vision Transformers (ViViT, ...)** | Tokenize space-time patches ("tubelets"), self-attention over all tokens | Very high (quadratic attention over space-time tokens) | Large datasets, pretraining available (transfer learning essentially required to make this practical) |
| **Video Transformers with factorized/windowed attention (VideoSwin, TimeSformer with divided space-time attention, ...)** | Attention factorized across space and time separately to control cost | High, but much better scaling than full joint attention | The current (2026) state-of-the-art default for large-scale video understanding when compute allows and pretrained weights are available |

**Practical guidance for a project of realistic scale** (hundreds to a few thousand labeled clips, not millions): start with a **pretrained 3D CNN or video transformer backbone (transfer learning), fine-tuned**, not a model trained from scratch. Video models are enormously data-hungry to train from scratch (Kinetics-scale pretraining uses hundreds of thousands of videos); nearly every practitioner in 2026 fine-tunes from a Kinetics- or large-scale-video-pretrained checkpoint (available via `torchvision.models.video`, `pytorchvideo`, or Hugging Face `transformers` video model hubs) rather than training from random initialization.

```python
# models/build_model.py
"""Transfer learning: pretrained video backbone, replaced head for the target task."""
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def build_video_classifier(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)   # pretrained on Kinetics-400

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False    # feature extraction mode — only the new head trains

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)   # new head always trainable, replaces the 400-class Kinetics head
    return model
```

**Fine-tuning vs. feature extraction trade-off**: freezing the backbone (feature extraction) trains fast and resists overfitting on small datasets but caps accuracy at what Kinetics-like features generalize to; unfreezing some or all layers (fine-tuning) can reach higher accuracy on domain-shifted data (e.g., surgical video looks very different from Kinetics' everyday-action videos) but needs more data and careful learning-rate scheduling (typically a much smaller LR on backbone layers than on the new head) to avoid destroying the pretrained features early in training.

---

## 6. Training Pipeline

```python
# train.py
"""
End-to-end training loop: mixed precision, gradient accumulation,
checkpointing, and experiment tracking hooks.
"""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path

CONFIG = {
    "num_epochs": 30,
    "lr": 1e-4,
    "backbone_lr": 1e-5,          # smaller LR for pretrained backbone layers
    "weight_decay": 1e-4,
    "grad_accum_steps": 4,         # effective batch size = loader batch_size * grad_accum_steps
    "grad_clip_norm": 1.0,
    "amp": True,
    "checkpoint_dir": "checkpoints/",
    "early_stop_patience": 5,
    "seed": 42,
}


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for step, (clips, labels) in enumerate(loader):
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # clips: [B, C, T, H, W]

        with autocast(device_type="cuda", enabled=config["amp"]):
            logits = model(clips)                    # [B, num_classes]
            loss = criterion(logits, labels) / config["grad_accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % config["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * config["grad_accum_steps"]

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        with autocast(device_type="cuda", enabled=True):
            logits = model(clips)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def save_checkpoint(state: dict, path: str):
    """Resume-safe: saves model, optimizer, scaler, epoch, and best-metric state."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_video_classifier(num_classes=7, freeze_backbone=False).to(device)

    # differential LR: backbone vs. new head
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = model.fc.parameters()
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": CONFIG["backbone_lr"]},
        {"params": head_params, "lr": CONFIG["lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    scaler = GradScaler(enabled=CONFIG["amp"])

    best_val_acc, patience_counter, start_epoch = 0.0, 0, 0

    # resume-safe checkpoint loading
    ckpt_path = Path(CONFIG["checkpoint_dir"]) / "last.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, CONFIG)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        save_checkpoint({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_val_acc": best_val_acc,
        }, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc, patience_counter = val_acc, 0
            save_checkpoint({"model": model.state_dict()}, Path(CONFIG["checkpoint_dir"]) / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
```

**Multi-GPU / distributed**: for video specifically, `DistributedDataParallel` (DDP) is preferred over `DataParallel` unconditionally — `DataParallel` replicates the model on every forward pass and gathers outputs on one GPU, which is especially painful for video's large activation memory (5D tensors). Launch with `torchrun --nproc_per_node=N train.py`, wrap the model in `DDP`, and use `DistributedSampler` so each GPU sees a distinct shard of videos (critical to also shard at the *video* level, not clip level, to preserve the leakage-safe split).

**Reproducibility vs. performance**: `torch.backends.cudnn.deterministic = True` + `torch.use_deterministic_algorithms(True)` give bit-reproducible runs but can slow 3D convolutions meaningfully (cuDNN's fastest 3D conv algorithms are often non-deterministic). Standard practice: use deterministic settings for debugging/final reported results, `cudnn.benchmark = True` (non-deterministic, auto-tunes the fastest kernel per input shape) for day-to-day experimentation where speed matters more than exact reproducibility.

---

## 7. Efficiency & Scalability

**Diagnosing the bottleneck** — this is the single highest-leverage skill for video training, since video pipelines bottleneck on CPU/IO far more often than image pipelines do:

1. Run `nvidia-smi -l 1` (or `nvtop`) during training. If GPU utilization is consistently <70-80%, the bottleneck is upstream of the GPU (data loading), not the model.
2. Use the PyTorch Profiler to confirm: `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])` — look at the ratio of time in `DataLoader.__next__` vs. actual forward/backward.
3. If data loading dominates: increase `num_workers`, check `pin_memory=True` (enables faster host→device transfer via page-locked memory), `persistent_workers=True` (avoids re-spawning worker processes every epoch — meaningful for video where each worker's decoder state has non-trivial setup cost), and `prefetch_factor` (how many batches each worker prepares ahead of time).
4. If GPU compute dominates (utilization near 100% but training is still slow): consider mixed precision (if not already on), a smaller/faster architecture, gradient checkpointing to allow a larger batch size, or multi-GPU.
5. If storage I/O dominates (common with videos on slow network filesystems, e.g., mounted cloud storage): pre-fetching, local SSD caching of the active shard, or switching to a sequential-read-friendly format (e.g., WebDataset-style sharded tar files) rather than thousands of small random-access file opens.

```python
# dataloader_config.py
"""Reasoned DataLoader configuration, not just 'more workers = faster'."""
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,                # video batches are memory-heavy; often smaller than image batch sizes
    shuffle=True,
    num_workers=8,                # rule of thumb: start at (CPU cores - 2), then profile and adjust
    pin_memory=True,              # speeds host->device copy; skip if training on CPU only
    persistent_workers=True,      # avoids worker restart overhead each epoch
    prefetch_factor=4,            # workers prepare 4 batches ahead — higher = more RAM, smoother pipeline
    drop_last=True,               # avoids a ragged final batch destabilizing BatchNorm statistics
)
```

**Reducing VRAM usage**:
- **Mixed precision (AMP)** — the default first step; roughly halves activation memory and often speeds up compute on modern GPUs with Tensor Cores.
- **Gradient checkpointing** — trades compute for memory by not storing all intermediate activations, recomputing them during backward. Video models with many temporal frames have activation memory that scales with `T`, making this disproportionately more valuable for video than for image models — often the difference between "doesn't fit" and "fits" on a single consumer GPU.
- **Smaller clip length / lower resolution** — the blunt-but-effective lever; halving `T` roughly halves activation memory for many architectures.
- **Gradient accumulation** — simulate a larger effective batch size without the memory cost of a large actual batch (shown in the training loop above).

```python
# enabling gradient checkpointing on a torchvision video model
model.stem.requires_grad_(True)
for i, block in enumerate(model.layer3):
    block = torch.utils.checkpoint.checkpoint_wrapper(block)  # wraps block to recompute activations in backward
```

**Caching decoded frames**: if disk space allows and your dataset is small enough, decoding once and caching to a fast format (e.g., resized frames as a memory-mapped array, or pre-extracted `.pt` tensors) removes repeated decode cost across epochs — a strong win when the dataset is small enough to fit but epochs are decode-bound, a much weaker win (or a net loss, due to disk space and cache-management complexity) on datasets too large to cache.

---

## 8. Advanced Techniques

- **Multi-clip / multi-view inference**: at test time, sample several clips per video (e.g., 3 temporal clips × 3 spatial crops = 9 "views") and average predictions. This is standard practice for reporting benchmark numbers (used in the original Kinetics/SlowFast papers) because a single random clip is a noisy sample of a whole video's content — it costs 9× inference compute for a real accuracy gain, so it's usually reserved for final evaluation/deployment rather than every training-time validation step.
- **Knowledge distillation**: train a smaller/faster "student" model to match a larger pretrained "student"'s soft outputs — relevant when you need a 3D-CNN-accuracy model that must run in real time (e.g., a lightweight model deployed for live intraoperative feedback) but can afford heavy compute for the teacher offline.
- **Quantization**: post-training INT8 quantization for inference speed on edge/embedded deployment; video models' 3D convolutions quantize less cleanly than 2D CNNs in some toolchains as of 2026, so validate accuracy drop empirically rather than assuming parity.
- **Parameter-efficient fine-tuning (LoRA, adapters)**: increasingly used for large pretrained video transformers, where full fine-tuning is prohibitively expensive; LoRA on attention projection layers is a reasonable default starting point.
- **Long videos**: don't try to fit an entire hour-long video through a video transformer at once (quadratic attention cost makes this infeasible). Standard approaches: (a) hierarchical models — encode short clips independently, then a lightweight temporal model over clip-level features for the full video; (b) sliding-window inference with overlap-averaging; (c) memory-augmented/streaming transformers that carry compressed state across windows.
- **Streaming/online inference**: for real-time applications, causal architectures (that only look at past frames, never future ones) are required — a standard offline-trained model that used bidirectional temporal context will not work correctly if naively deployed frame-by-frame, since it implicitly assumes access to future frames within its clip window.

---

## 9. Evaluation

**Metric choice must match label granularity** (this is where video evaluation most often goes wrong):

| Task | Primary metrics | Notes |
|---|---|---|
| Clip/video classification | Accuracy, top-k accuracy, per-class F1/precision/recall, confusion matrix | Top-5 is a Kinetics-era convention; for a small number of classes (e.g., 7 surgical phases) top-1 with per-class breakdown is more informative |
| Multi-label video tagging | mAP (mean average precision) | Standard accuracy doesn't apply when multiple labels can be true simultaneously |
| Temporal event detection | Temporal IoU-based mAP (predicted interval vs. ground-truth interval overlap) | Analogous to object detection mAP but over the time axis instead of image space |
| Frame-level dense prediction | Per-frame accuracy, but also **edit distance / segmental metrics** for phase recognition specifically | Per-frame accuracy alone rewards "always predict the majority phase" and hides boundary errors; segmental F1 (correct if the predicted segment overlaps sufficiently with ground truth) is standard in surgical phase recognition literature specifically |

**Confusion matrices and per-class analysis** matter more in video classification than raw accuracy suggests, especially with imbalanced clinical classes — a model can hit 92% accuracy by nailing the common "routine dissection" phase and never correctly identifying the rare "complication" phase, which is exactly the failure mode that matters most in a clinical deployment context.

**Leakage-safe evaluation checklist** before trusting any reported number:
- Are train/val/test split at the video (or patient/session) level, not clip level? *(Section 2)*
- Was the test set touched at all during model/hyperparameter selection? (Val, not test, should drive early stopping and hyperparameter choices.)
- If multi-clip inference is used at test time, is it applied consistently across all models being compared?
- Are class-imbalanced metrics (macro-F1, per-class recall) reported alongside accuracy, not instead of it, but not omitted either?

---

## 10. Complete Practical Project — Structure

A minimal but complete, runnable project layout tying everything above together:

```
surgical_phase_project/
├── data/
│   ├── videos/                     # raw .mp4 files
│   ├── annotations/{train,val,test}.csv
│   └── metadata.json
├── datasets/
│   └── surgical_video_dataset.py   # Section 3
├── transforms/
│   └── video_transforms.py         # Section 4
├── models/
│   └── build_model.py              # Section 5
├── splits/
│   └── make_splits.py              # Section 2
├── train.py                        # Section 6
├── evaluate.py                     # confusion matrix, per-class F1, segmental metrics
├── inference.py                    # multi-clip inference + Grad-CAM-style visualization on sampled frames
├── profiling/
│   └── profile_pipeline.py         # torch.profiler harness, Section 7
├── checkpoints/
├── configs/
│   └── default.yaml                # externalizes CONFIG dict for experiment tracking
└── requirements.txt
```

`evaluate.py` (key part — confusion matrix + per-class metrics, matching Section 9):

```python
# evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

@torch.no_grad()
def evaluate_model(model, loader, device, class_names, num_views: int = 1):
    """num_views > 1 triggers multi-clip averaging (Section 8)."""
    model.eval()
    all_preds, all_labels = [], []

    for clips, labels in loader:
        # clips: [B, C, T, H, W] or [B, num_views, C, T, H, W] if multi-view sampling upstream
        clips = clips.to(device)
        if num_views > 1:
            B, V = clips.shape[:2]
            logits = model(clips.view(B * V, *clips.shape[2:]))
            logits = logits.view(B, V, -1).mean(dim=1)   # average softmax-space or logit-space views
        else:
            logits = model(clips)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_preds, all_labels
```

---

## 11. Best Practices vs. Outdated Techniques

**Fundamentals (stable, unlikely to change):**
- Split at the video/patient level, never the clip level.
- Apply spatial augmentation consistently across all frames in a clip.
- Mixed precision as a near-default for GPU training.
- Group-aware splitting whenever a confounding grouping variable exists.

**Current recommended practice (2026):**
- Transfer learning from large-scale video-pretrained backbones over training from scratch, for nearly all practically-sized datasets.
- Decord (or DALI at scale) over OpenCV for the primary decode path.
- Factorized/windowed-attention video transformers or SlowFast-style 3D CNNs as strong defaults, chosen based on available compute and pretrained-weight availability.
- `DistributedDataParallel` over `DataParallel` for any multi-GPU setup.
- Segmental/temporal-aware metrics (not just frame accuracy) for phase/action-segmentation tasks.

**Research-frontier / not yet default:**
- Fully learned, data-driven frame-sampling policies (learning *which* frames to sample, not just how many).
- Long-video streaming transformers with compressed memory tokens.
- Video-language pretraining (contrastive video-text models) as a general-purpose feature source, in domains (like clinical video) where large paired text descriptions are scarce, so its benefit over Kinetics-style pretraining is domain-dependent and not yet a safe default.

**Outdated / generally discouraged:**
- Two-stream optical-flow architectures as a first choice — flow computation is expensive and largely matched or exceeded by 3D CNNs/transformers that learn motion implicitly; still occasionally useful but no longer the default starting point it was pre-2020.
- Frame-by-frame independent augmentation (destroys temporal coherence).
- `DataParallel` for multi-GPU video training (memory-inefficient relative to DDP).
- Training large video models entirely from scratch on small, domain-specific datasets rather than fine-tuning — this reliably underperforms transfer learning at realistic clinical-dataset scales and wastes compute.

---

## 12. Hardware-Dependent Pipeline Changes

- **CPU-only**: feasible only for very small models/datasets or feature extraction with a frozen, small backbone; expect decode + preprocessing to dominate wall-clock time. Favor 2D-CNN-per-frame + temporal pooling over 3D CNNs (much lower compute), and keep clip length and resolution small.
- **Entry-level GPU (e.g., 6–8GB VRAM)**: mixed precision and gradient checkpointing are close to mandatory, not optional, for 3D CNNs or transformers at reasonable resolution/clip-length. Favor smaller backbones (e.g., R(2+1)D-18/34 over R3D-152-scale models) and shorter clips (8–16 frames).
- **High-end consumer GPU (e.g., 24GB VRAM)**: comfortable range for fine-tuning mid-sized 3D CNNs/video transformers at moderate resolution and batch size without checkpointing; still worth profiling before assuming compute (rather than data loading) is the bottleneck.
- **Multi-GPU workstation**: DDP becomes worthwhile once a single epoch is slow enough that wall-clock time — not just VRAM — is the constraint; also useful for running larger effective batch sizes for architectures sensitive to batch size (e.g., contrastive pretraining objectives).
- **Cloud GPU infrastructure**: storage I/O (network-mounted data) frequently becomes the bottleneck even when it wasn't locally — budget for local NVMe caching of the active data shard, or a sharded/streaming format designed for sequential reads (WebDataset-style), rather than assuming cloud storage bandwidth matches local SSD.

---

## 13. Learning Roadmap

**Prerequisites** (you likely already have most of this, given your existing CNN/transformer curriculum): solid PyTorch `Dataset`/`DataLoader`/training-loop fluency; 2D CNN transfer learning (already done); basic understanding of attention/transformers (already done via your GPT-2-from-scratch work).

**Progressive project sequence:**
1. **Single-frame baseline**: classify individual frames from short clips with a 2D CNN (e.g., your existing ResNet-50 pipeline, applied per-frame) — establishes a non-temporal baseline to know what temporal modeling actually buys you.
2. **Clip-level 3D CNN fine-tuning**: fine-tune a pretrained `r3d_18` (as above) on a small clip-classification dataset — get the full Section 3–6 pipeline working end-to-end on something manageable (e.g., a public action-recognition subset, or a small self-collected clinical video set).
3. **Add proper evaluation + leakage-safe splitting**: retrofit group-aware splitting and per-class/confusion-matrix evaluation onto project 2 — this is as important a skill as the modeling itself.
4. **Temporal phase/segmentation task**: move from clip classification to frame-level or segment-level phase recognition (the harder, more clinically realistic task) — requires a different loss/head design and segmental evaluation metrics.
5. **Efficiency pass**: profile project 4's pipeline, identify the actual bottleneck (Section 7), and fix it — deliberately practicing bottleneck diagnosis rather than only reading about it.
6. **Video transformer fine-tuning**: repeat project 2/4 with a pretrained video transformer (e.g., VideoSwin or a Hugging Face video model) instead of a 3D CNN, and compare accuracy/compute trade-offs directly — builds the comparative intuition that's hard to get from reading alone.
7. **Multi-GPU/distributed training**: scale project 6 across multiple GPUs with DDP, even if you only have access to 2, to build the operational muscle memory before you need it at larger scale.
8. **Deployment-oriented capstone**: package a trained phase-recognition model for near-real-time inference (streaming-compatible architecture or sliding window), including quantization and a latency/accuracy trade-off analysis — this is the project that most directly demonstrates the "biomedical + deployment" differentiator you're building toward.

**Key references** (verify current versions/links, as APIs move):
- PyTorch official video docs: `torchvision.models.video`, `torchvision.io` — https://docs.pytorch.org/vision/stable/models.html#video-classification
- PyTorchVideo (Meta AI's video-specific library, model zoo + transforms): https://pytorchvideo.org/
- Decord: https://github.com/dmlc/decord
- SlowFast paper (Feichtenhofer et al., 2019): https://arxiv.org/abs/1812.03982
- ViViT paper (Arnab et al., 2021): https://arxiv.org/abs/2103.15691
- Video Swin Transformer (Liu et al., 2022): https://arxiv.org/abs/2106.13230
- NVIDIA DALI docs (GPU-accelerated data loading): https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
- PyTorch Profiler docs: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Kinetics dataset / benchmark conventions: https://www.deepmind.com/open-source/kinetics
