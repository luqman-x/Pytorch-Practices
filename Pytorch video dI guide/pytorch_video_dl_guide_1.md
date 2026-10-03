# Training Video AI Models in PyTorch — Simple Guide (2026)

_The examples below use surgery videos as the topic, since that fits your biomedical interests. But the same code works for any kind of video, like sports or everyday actions._

---

## Table of Contents

1. Video Basics
2. Organizing Your Dataset
3. Loading Videos in PyTorch
4. Preparing Videos for the Model
5. Choosing a Model Type
6. The Training Loop
7. Making It Fast
8. Advanced Tricks
9. Checking If the Model Is Good
10. Full Project Example
11. What's Good vs. Outdated
12. What to Use Based on Your Computer
13. Your Learning Path

---

## 1. Video Basics

**What is a video, really?** A video is just a stack of pictures (called **frames**) shown quickly one after another. **FPS** means "frames per second" — how many pictures flash by every second. The computer doesn't care about the video file itself; what matters is the numbers (called a **tensor**) that come out after the video is unpacked: shape `[T, C, H, W]` for one clip, meaning Time (how many frames), Color channels (3 for red/green/blue), Height, and Width.

- **Container vs. codec** — Think of a video file like a shipping box (**container**, e.g. `.mp4`) holding compressed goods inside (**codec**, e.g. `H.264`). The codec is the trick used to squeeze the video down so it isn't huge. `H.264` is the most common and almost every computer/GPU can unpack it fast. Some newer codecs (like `AV1`) save more space but are slower to unpack unless your hardware is new. Since you'll be unpacking videos millions of times during training, picking a codec that's fast to unpack really matters.
- **Why frames aren't treated like separate photos** — Frames right next to each other look almost the same (that's why video "flows" smoothly). If you just looked at each frame alone like a photo, you'd miss all the _motion_ — which is often the whole point. That's why video AI needs special tricks beyond normal photo AI.
- **Key frames** — Compressed video doesn't store every frame as a full picture. It stores a few full pictures ("key frames") and then just the _changes_ for the frames in between, like a flipbook that only redraws what moved. This means jumping to a totally random frame in the middle of a video is slow — the computer has to rebuild it starting from the last key frame. So instead, we usually grab a _chunk_ of nearby frames at once ("a clip"), which is much faster.

**What does the label (answer) attach to?** This changes what kind of problem you're solving:

| Task                     | Label is attached to    | Example                                        |
| ------------------------ | ----------------------- | ---------------------------------------------- |
| Video classification     | The whole clip          | "This video shows a gallbladder surgery"       |
| Action/phase recognition | A time segment          | "Cutting phase: 4:12 to 9:47"                  |
| Event detection          | One moment in time      | "Clip applied at 7:03"                         |
| Object tracking          | Every frame, with a box | Where the surgical tool tip is, frame by frame |
| Frame-by-frame labeling  | Literally every frame   | What phase is happening at each exact frame    |

The first two are pretty similar — both just want you to name "what's happening." The last three need the model to say _where_ and _when_, which is a harder, different kind of problem with different tools.

**Label files (annotations)** — For simple "this clip = this label" tasks, you just need a spreadsheet (CSV) or text file (JSON) matching a video name to its label. For time-based tasks, you need a start time, end time, and label for each segment. Different public datasets format this differently, so you'll usually write a small script to convert whatever format you're given into the format your code expects.

---

## 2. Organizing Your Dataset

Keep your original video files separate from anything you generate later (like saved frames or index files), so you can always regenerate the extra stuff without messing up your source videos:

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
├── metadata.json         # fps, length, size, patient/session ID for each video
└── splits/
    └── split_seed42.json
```

**The #1 mistake people make: "data leakage."** This is when information from your test sneaks into your training by accident, making your model _look_ smart when it's actually just cheating or memorizing. In video, there are two sneaky ways this happens:

1. **Cutting the same video into pieces, then putting some pieces in training and some in testing.** Since neighboring clips from the same video look almost identical, the model basically "sees the answer" during testing because it already saw a nearly-identical clip during training. **Fix: split by whole video (or patient), never by individual clips.**
2. **Videos from the same patient/surgeon/camera ending up split between training and testing.** The model might learn to recognize _that specific camera's lighting_ instead of the actual medical task — then it "cheats" by spotting the camera, not the content. **Fix: group all videos from the same patient together when splitting, so a patient is either entirely in training or entirely in testing, never both.**

```python
# splits/make_splits.py
"""Splits data by patient group so no patient appears in two different splits."""
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

def make_group_split(metadata_csv: str, group_col: str = "patient_id", seed: int = 42):
    df = pd.read_csv(metadata_csv)  # columns: video_id, patient_id, label, ...
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df[group_col]))
    train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

    # split the leftover 15% into val and test, still keeping patients grouped together
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[group_col]))
    val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

    # double-check: no patient should be in two groups at once
    assert set(train_df[group_col]) & set(val_df[group_col]) == set()
    assert set(train_df[group_col]) & set(test_df[group_col]) == set()
    return train_df, val_df, test_df
```

**Uneven data (class imbalance)** — Say you have tons of "normal" video but very few "rare complication" clips. The model might just learn to always guess "normal" and still score well, while being useless at spotting the rare thing you actually care about. Fixes, easiest first:

- **Show rare examples more often** during training (`WeightedRandomSampler`) — simplest fix, try this first.
- **Make mistakes on rare examples "cost more"** in the loss calculation (`class weights`) — use this _together_ with the above, not instead of it.
- **Focal loss** — a special loss function that forces the model to pay extra attention to hard/rare examples. Use this if the imbalance is really extreme.
- **Grab more overlapping clips around the rare moments** — since video lets you sample clips anywhere, you can literally create more training examples out of the same rare event by sampling it from slightly different starting points.

**Check your files before training, not during:**

```python
import decord
from pathlib import Path

def audit_videos(video_dir: str) -> list[str]:
    """Returns a list of video files that are broken or unreadable."""
    bad = []
    for path in Path(video_dir).glob("*.mp4"):
        try:
            vr = decord.VideoReader(str(path))
            _ = len(vr)     # try reading basic info
            _ = vr[0]       # try reading the very first frame
        except Exception:
            bad.append(str(path))
    return bad
```

**Being able to repeat your results.** Write down and save: which seed number you used for splitting, the exact version of your annotation file, the exact versions of your Python libraries, and exactly how you're sampling clips (how many frames, how far apart). If you don't pin these down, running your "same" experiment twice can give different results — especially with smaller datasets — and you won't know if a change you made actually helped or if it was just random luck.

---

## 3. Loading Videos in PyTorch

**Which tool should unpack your videos?** (This list changes year to year — double check what's fastest when you actually build this, but here's the general picture for 2026):

| Tool                                | Speed    | Uses GPU to unpack?       | How easy                        | Best for                                                        |
| ----------------------------------- | -------- | ------------------------- | ------------------------------- | --------------------------------------------------------------- |
| OpenCV (`cv2`)                      | Slow-ish | No                        | Very easy                       | Learning, small projects, quick tests                           |
| torchvision's built-in video reader | Medium   | Sometimes                 | Easy, works nicely with PyTorch | Small-to-medium projects                                        |
| **Decord**                          | Fast     | Optional                  | Medium                          | **Best general pick for 2026 — good balance of speed and ease** |
| PyAV                                | Fast     | Manual setup              | More code needed                | When you need very specific control                             |
| NVIDIA DALI                         | Fastest  | Yes, full pipeline on GPU | Harder to set up                | Big projects training on lots of GPUs                           |

**Why this actually matters:** Your GPU (the part doing the heavy math) can often process a batch _faster_ than your CPU can unpack and prepare the next batch of video. If unpacking is too slow, your expensive GPU just sits there waiting — wasting time and money. This is the #1 speed problem in video AI projects, way more common than in photo AI projects, simply because video takes much more work to unpack than a single photo.

**A custom Dataset class that grabs clips:**

```python
# datasets/surgical_video_dataset.py
"""
A Dataset that pulls out a short clip from each video.

Shape of the data as it moves through:
  clip:  [T, H, W, C]  (raw numbers 0-255)   -> after processing: [C, T, H, W]  (decimal numbers)
"""
import decord
import numpy as np
import pandas as pd
import torch
from decord import VideoReader
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")  # makes decord hand back PyTorch tensors directly (saves an extra conversion step)


class SurgicalPhaseClipDataset(Dataset):
    def __init__(
        self,
        annotations_csv: str,
        video_dir: str,
        num_frames: int = 16,
        sampling: str = "uniform",     # "uniform" | "random" | "dense"
        frame_stride: int = 2,          # how many frames to skip, used by "dense" sampling
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
        """Decides WHICH frame numbers to grab from the video."""
        span = max(end - start, self.num_frames)
        if self.sampling == "uniform":
            # evenly spaced frames, always the same — good for testing (no randomness)
            return np.linspace(start, start + span - 1, self.num_frames).astype(int)
        elif self.sampling == "random":
            # slightly randomized spacing — good for training (adds variety)
            bin_size = span / self.num_frames
            offsets = np.random.uniform(0, bin_size, self.num_frames) if self.train else np.zeros(self.num_frames)
            return (start + np.arange(self.num_frames) * bin_size + offsets).astype(int)
        elif self.sampling == "dense":
            # grabs frames one after another (with gaps) — good when fine motion detail matters
            return np.arange(start, start + self.num_frames * self.frame_stride, self.frame_stride)
        raise ValueError(f"Unknown sampling: {self.sampling}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        vr = VideoReader(f"{self.video_dir}/{row.video_id}.mp4", num_threads=1)  # explained below
        indices = np.clip(self._sample_indices(row.start_frame, row.end_frame), 0, len(vr) - 1)

        clip = vr.get_batch(indices)              # [T, H, W, C] raw numbers
        clip = clip.permute(3, 0, 1, 2).float()    # reorder to [C, T, H, W] — the shape most models expect

        if self.transform:
            clip = self.transform(clip)

        label = self.label_to_idx[row.label]
        return clip, label
```

**Why `num_threads=1` here?** PyTorch already runs several "worker" processes in parallel for you (set by `num_workers` below). If each worker _also_ tries to use multiple threads internally, they end up competing for the same CPU cores and everything actually gets _slower_. Let PyTorch's workers handle the parallelism instead.

---

## 4. Preparing Videos for the Model

**The standard shape** models expect is `[Batch, Time, Channels, Height, Width]` — but different libraries sometimes order Time and Channels differently, so always double-check what shape the specific model you're using wants. Getting this wrong is sneaky: your code won't crash, it will just train a model on scrambled, meaningless data, and you won't find out until your results look oddly bad.

**Two kinds of "data augmentation" (making small random changes to your data to help the model generalize better):**

- **Spatial changes** (cropping, flipping, color changes) — these must be applied **the exact same way to every frame in a clip**. If you crop frame 1 differently than frame 2, you've destroyed the motion information the model needs — it would look like things are randomly jumping around.
- **Time-based changes** — randomizing where a clip starts, slightly shifting which frames you pick, skipping frames to simulate faster motion, or occasionally dropping a frame. One trick to be careful with: **reversing time**. For some actions (like "pushing" vs. "pulling"), playing it backward actually changes what it means — so don't use this trick blindly.

```python
# transforms/video_transforms.py
"""Spatial changes that apply identically to every frame in the clip."""
import torch


class ClipRandomCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # clip shape: [C, T, H, W]
        _, _, h, w = clip.shape
        top = torch.randint(0, max(h - self.size, 1), (1,)).item()
        left = torch.randint(0, max(w - self.size, 1), (1,)).item()
        return clip[:, :, top:top + self.size, left:left + self.size]   # same crop box used on every frame


class ClipRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p   # probability of flipping

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return clip.flip(dims=[-1])    # flips every frame the same way
        return clip


class ClipNormalize:
    """Rescales pixel values to a standard range models expect (roughly -1 to 1)."""
    def __init__(self, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip / 255.0 - self.mean) / self.std
```

**Videos of different lengths** — Instead of stretching or padding every video to match a fixed length (wasteful), it's usually better to just grab a fixed-size _clip_ from anywhere in the video, as shown above. Padding is really only needed when your task requires seeing the _whole_ video at once.

**Don't waste time thinking resolution saves decoding time** — unpacking a video takes about the same effort no matter what size you plan to shrink it to afterward. The real time-savers are: shrinking the image size _right after_ unpacking (before doing anything else with it), and simply grabbing fewer frames per clip. If you're going to train on the same dataset many times, it can also be worth resizing and saving all your videos once, ahead of time, so every future training run skips that step.

---

## 5. Choosing a Model Type

| Type                                                                                            | How it works (simply)                                                               | How heavy                                                                            | When to use it                                                                                                       |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **CNN + simple time-combiner** (e.g. a photo-AI model run on each frame, then results combined) | Looks at each frame like a photo, then stitches the results together                | Light                                                                                | Small datasets, limited computer power, or when exact motion isn't super important (e.g. "is there a tool in frame") |
| **3D CNNs** (C3D, I3D, SlowFast)                                                                | Looks at space _and_ time together, like a cube instead of flat photos              | Heavy                                                                                | When motion really matters; a good balanced default                                                                  |
| **ConvLSTM / "memory" models**                                                                  | Processes frame by frame while remembering what came before                         | Medium, but must go one frame at a time (can't speed up by doing frames in parallel) | Live/real-time use, where you need an answer _while_ the video is still playing                                      |
| **Two-stream (color + motion)**                                                                 | One AI looks at colors, another looks only at movement patterns, results combined   | Heavy (computing the "motion patterns" part is itself slow)                          | Older approach; mostly replaced now by 3D CNNs and transformers that learn motion on their own                       |
| **Video Vision Transformers**                                                                   | Breaks the video into small 3D chunks and compares every chunk to every other chunk | Very heavy                                                                           | Works best with huge datasets and a model that's already pre-trained                                                 |
| **Modern video transformers (smarter attention)**                                               | Same idea as above but smarter about which chunks to compare, saving compute        | Heavy, but much more efficient                                                       | The strongest choice in 2026 if you have decent compute and a pretrained model to start from                         |

**The practical advice for a real project** (hundreds to a few thousand clips, which is realistic for most people — not millions): **start from a model that's already been trained on a huge public video dataset, and just fine-tune it for your task**, rather than training completely from scratch. Training from zero needs an enormous amount of data that almost nobody has. Starting from a pretrained model and adjusting it is what almost everyone actually does.

```python
# models/build_model.py
"""Starts from a model already trained on a big public video dataset, swaps the final layer for our task."""
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def build_video_classifier(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)   # pretrained on a huge public video dataset

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False    # "lock" the pretrained part so it doesn't change — only train the new layer

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)   # swap the final layer for our own set of labels
    return model
```

**Should you "lock" the pretrained part or let it adjust too?** Locking it (feature extraction) trains fast and is safer with small datasets, but your accuracy has a ceiling. Letting it adjust (fine-tuning) can get better results, especially when your videos look very different from the public dataset it was trained on (e.g., surgery videos look nothing like everyday action videos) — but needs more data and a gentler touch (a smaller learning rate on the older layers) so you don't accidentally wreck what it already learned.

---

## 6. The Training Loop

```python
# train.py
"""
The main training script. Includes: faster math (mixed precision),
pretending we have a bigger batch than we do (gradient accumulation),
and saving progress so we can resume if interrupted (checkpointing).
"""
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path

CONFIG = {
    "num_epochs": 30,
    "lr": 1e-4,                   # learning rate for the new layer
    "backbone_lr": 1e-5,          # smaller learning rate for the pretrained part
    "weight_decay": 1e-4,
    "grad_accum_steps": 4,         # pretend our batch is 4x bigger than it really is
    "grad_clip_norm": 1.0,         # stops updates from being too extreme
    "amp": True,                   # use faster, lower-precision math
    "checkpoint_dir": "checkpoints/",
    "early_stop_patience": 5,      # stop early if no improvement for 5 epochs
    "seed": 42,
}


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for step, (clips, labels) in enumerate(loader):
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # clips shape: [B, C, T, H, W]

        with autocast(device_type="cuda", enabled=config["amp"]):
            logits = model(clips)                    # the model's guesses, shape [B, num_classes]
            loss = criterion(logits, labels) / config["grad_accum_steps"]

        scaler.scale(loss).backward()  # calculates how to adjust the model

        if (step + 1) % config["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)     # actually updates the model
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * config["grad_accum_steps"]

    return running_loss / len(loader)


@torch.no_grad()   # don't calculate updates here, we're just checking performance
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
    """Saves everything needed to pick back up later if training is interrupted."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    torch.manual_seed(CONFIG["seed"])  # makes randomness repeatable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_video_classifier(num_classes=7, freeze_backbone=False).to(device)

    # use a smaller learning rate for the pretrained part, bigger for the new layer
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = model.fc.parameters()
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": CONFIG["backbone_lr"]},
        {"params": head_params, "lr": CONFIG["lr"]},
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    scaler = GradScaler(enabled=CONFIG["amp"])

    best_val_acc, patience_counter, start_epoch = 0.0, 0, 0

    # if a saved checkpoint exists, pick up where we left off
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

**Training on more than one GPU at once:** use `DistributedDataParallel` (DDP), not the older `DataParallel` — DDP is much more memory-efficient, and video data already uses a lot of memory, so this matters more for video than for photos. Each GPU should get a different _set of whole videos_ (not clips) to avoid the leakage problem from Section 2.

**Repeatable vs. fast:** You can force PyTorch to give the exact same result every single run (`deterministic` settings), but this can slow things down. Most people use exact-repeatability only when debugging or reporting final results, and use the faster (slightly less repeatable) settings for everyday experimenting.

---

## 7. Making It Fast

**How to figure out what's actually slowing you down** — this is one of the most useful skills for video AI, because the slow part is usually _not_ what people assume:

1. Watch your GPU usage while training (`nvidia-smi -l 1`). If it's below about 70-80% busy most of the time, your GPU is sitting around waiting — the slowdown is happening _before_ the GPU gets the data (i.e., during unpacking/preparing the video).
2. Use PyTorch's built-in Profiler tool to confirm exactly where time is going.
3. If data loading is the slow part: increase `num_workers` (more parallel helpers preparing data), turn on `pin_memory=True` (speeds up sending data to the GPU), turn on `persistent_workers=True` (avoids restarting helpers every epoch, which is costly for video), and raise `prefetch_factor` (how many batches get prepared in advance).
4. If the GPU itself is the slow part (it's busy ~100% the whole time): try faster math (mixed precision), a lighter model, gradient checkpointing (explained below) to allow bigger batches, or more GPUs.
5. If reading files off disk/network is slow (common with cloud storage): cache files locally on a fast drive, or switch to a storage format built for fast sequential reading instead of many small random file opens.

```python
# dataloader_config.py
"""A well-reasoned DataLoader setup — not just cranking every number up."""
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,                 # video takes a lot of memory, so batches are often smaller than for photos
    shuffle=True,
    num_workers=8,                 # good starting point: (number of CPU cores) minus 2, then adjust after testing
    pin_memory=True,               # speeds up sending data to GPU (skip this if using CPU only)
    persistent_workers=True,       # keeps helper processes alive between epochs instead of restarting them
    prefetch_factor=4,             # how many batches get prepared ahead of time
    drop_last=True,                # drops a weirdly-sized last batch so it doesn't mess up training stats
)
```

**Running out of GPU memory? Here's the fix order:**

- **Mixed precision (faster, lower-precision math)** — try this first; usually cuts memory roughly in half with little downside.
- **Gradient checkpointing** — trades a bit of speed for a lot of memory savings, by not storing every intermediate calculation and instead recalculating some of them when needed. Very useful for video, since video uses way more memory per step than photos do.
- **Fewer frames per clip or smaller image size** — the simple, blunt fix. Cutting your frame count in half roughly cuts memory use in half too.
- **Gradient accumulation** — lets you simulate a bigger batch size without needing the memory of an actual bigger batch (already shown in the training loop above).

```python
# turning on gradient checkpointing for part of a model
model.stem.requires_grad_(True)
for i, block in enumerate(model.layer3):
    block = torch.utils.checkpoint.checkpoint_wrapper(block)  # saves memory by recalculating instead of storing
```

**Saving unpacked frames so you don't redo the work every time** — if your dataset is small enough to fit on disk in its unpacked form, you can save time by unpacking once and reusing it for every future training run, instead of unpacking the same video over and over every epoch. If your dataset is too big to fit, this trick stops being worth the hassle.

---

## 8. Advanced Tricks

- **Testing with multiple clips per video and averaging the answers** — Instead of grabbing just one random clip from a video at test time, grab several (e.g., from different moments and slightly different crops) and average the predictions. This gives a more reliable final answer, at the cost of needing to run the model several times per video — so it's usually only done for final testing, not during every regular check-in while training.
- **Knowledge distillation** — Train a big, accurate but slow model first, then train a small, fast model to copy its answers. Useful when you need something quick enough to run live (e.g., giving real-time feedback during surgery) but still want it to be nearly as smart as the big model.
- **Quantization** — Shrinking the model's numbers down to a simpler format so it runs faster, usually for deploying it on smaller devices. Double-check your accuracy doesn't drop too much afterward — video models don't always handle this as cleanly as photo models do.
- **Parameter-efficient fine-tuning (like LoRA)** — Instead of adjusting the _entire_ pretrained model (expensive), you only adjust a small added piece. This saves a lot of compute, especially useful for very large pretrained models.
- **Handling really long videos** (like an hour-long surgery) — You generally can't feed an entire long video into the model at once; it's too much. Common fixes: break it into short clips, process each one, then combine the results with a lighter second model; or slide a window across the video a bit at a time.
- **Watching video live, as it happens** — If the model needs to react in real time, it can only look at _past_ frames, not future ones (since future frames haven't happened yet!). A model trained with access to "future" frames within a clip won't work correctly if you just drop it into a live, real-time setting — this needs a different kind of model design from the start.

---

## 9. Checking If the Model Is Good

**Pick the right way to measure "good" based on your task** — this is where people most often get fooled by their own results:

| Task                             | How to measure it                               | Why                                                                                                                                                       |
| -------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clip/video classification        | Accuracy, per-class scores, confusion matrix    | With few classes (e.g. 7 surgical phases), breaking down performance per class tells you more than one overall number                                     |
| Multiple labels at once          | mAP (mean average precision)                    | Regular accuracy doesn't make sense when more than one label can be true at the same time                                                                 |
| Finding specific moments in time | Time-based overlap scoring                      | Similar to scoring how well a predicted box overlaps a real box, but for time instead of space                                                            |
| Labeling every single frame      | Per-frame accuracy _plus_ segment-based scoring | Per-frame accuracy alone can be tricked — a model that just always guesses the most common phase can still score high while being useless for rare phases |

**Why breaking results down by category matters extra in your field** — Imagine a model gets 92% accuracy overall, but it does that just by nailing the common "routine" phase and completely failing to recognize the rare "something went wrong" phase. The overall score looks great, but the model is useless for exactly the moment that matters most clinically. Always look at performance _per category_, not just the overall number.

**Before trusting any result, double-check:**

- Did you split by whole video/patient, not by individual clip? (Section 2)
- Was the test set kept completely untouched until the very end — never peeked at while tuning the model?
- If using multiple clips per video at test time, did you do this the same way for every model you're comparing?
- Are you also reporting per-category scores, not relying on one overall number alone?

---

## 10. Full Project Example — Structure

A simple, complete project layout that ties everything above together:

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
├── evaluate.py                     # scoring and breakdown by category
├── inference.py                    # running the trained model, with visualization
├── profiling/
│   └── profile_pipeline.py         # finding slow spots, Section 7
├── checkpoints/
├── configs/
│   └── default.yaml                # all the settings in one place
└── requirements.txt
```

`evaluate.py` (the key scoring part, matching Section 9):

```python
# evaluate.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

@torch.no_grad()
def evaluate_model(model, loader, device, class_names, num_views: int = 1):
    """num_views > 1 means we grab several clips per video and average the guesses (Section 8)."""
    model.eval()
    all_preds, all_labels = [], []

    for clips, labels in loader:
        # clips shape: [B, C, T, H, W] or [B, num_views, C, T, H, W] if sampling multiple views
        clips = clips.to(device)
        if num_views > 1:
            B, V = clips.shape[:2]
            logits = model(clips.view(B * V, *clips.shape[2:]))
            logits = logits.view(B, V, -1).mean(dim=1)   # average the guesses from all views
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

## 11. What's Good vs. Outdated

**The basics (won't really change over time):**

- Split by whole video/patient, never by individual clip.
- Apply the exact same crop/flip to every frame in a clip.
- Use mixed precision (faster math) by default when training on a GPU.
- Group patients together when splitting, whenever that grouping could leak information.

**What's recommended right now (2026):**

- Start from a pretrained video model and fine-tune it, rather than training from scratch, for almost any realistically-sized project.
- Use Decord (or DALI for very large-scale training) rather than OpenCV for unpacking videos.
- Use modern smart-attention video transformers or SlowFast-style 3D CNNs as strong default choices.
- Use `DistributedDataParallel`, not the older `DataParallel`, for multi-GPU training.
- Use segment-based scoring, not just per-frame accuracy, for phase/action tasks.

**Cutting-edge, not yet the "safe default":**

- AI that learns _which_ frames to look at, instead of us deciding that manually.
- "Streaming" transformers for very long videos that carry forward a compressed memory.
- Models pretrained on paired video + text — promising, but for specialized fields like clinical video (where there isn't much paired text data), it's not yet a guaranteed win over regular video pretraining.

**Outdated / usually not worth using anymore:**

- Two-stream (separate color + motion) models as your first choice — mostly replaced by models that learn motion on their own.
- Applying different random changes to each frame separately (breaks the motion information).
- `DataParallel` for multi-GPU training (uses memory inefficiently compared to DDP).
- Training a big video model completely from scratch on a small, specialized dataset — this almost always does worse than fine-tuning a pretrained model, and wastes a lot of compute.

---

## 12. What to Use Based on Your Computer

- **No GPU (CPU only):** Only realistic for small models/datasets, or just extracting features with a locked, small model. Expect most of your time to go to unpacking/preparing data. Favor simpler photo-AI-per-frame approaches over heavy 3D models, and keep clips short and small.
- **Entry-level GPU (6–8GB memory):** Mixed precision and gradient checkpointing aren't optional — you'll likely need both. Favor smaller models and shorter clips (8–16 frames).
- **High-end consumer GPU (24GB memory):** Comfortable for fine-tuning mid-sized models at decent quality without needing checkpointing. Still worth double-checking whether your GPU or your data loading is the actual bottleneck before assuming.
- **Multiple GPUs in one computer:** Worth setting up once a single training run takes long enough that time (not just memory) becomes the real constraint.
- **Cloud GPUs:** Reading files over the network often becomes the new bottleneck, even if it wasn't a problem locally — plan to cache data on a fast local drive, or use a storage format built for fast sequential reading.

---

## 13. Your Learning Path

**What you probably already know:** comfortable with PyTorch's Dataset/DataLoader and training loops; already done 2D CNN fine-tuning; already have a basic understanding of attention/transformers from your GPT-2-from-scratch work. That puts you in a good spot to start.

**Suggested order of projects, easiest to hardest:**

1. **Single-frame baseline** — classify individual frames using a normal 2D photo-AI model (like your existing ResNet-50 setup), one frame at a time, no video-specific tricks yet. This tells you how much _motion_ actually helps once you add it later.
2. **Basic 3D CNN fine-tuning** — fine-tune a pretrained `r3d_18` model on a small clip-classification dataset, getting the whole Sections 3–6 pipeline running end-to-end on something manageable.
3. **Add proper, leakage-safe evaluation** — go back to project 2 and add patient/video-level splitting plus per-category scoring — just as important a skill as the modeling itself.
4. **Phase/segment recognition task** — move up to labeling actual time segments or frames (the harder, more clinically realistic version) — this needs a different setup and segment-based scoring.
5. **Speed/efficiency pass** — profile project 4, figure out the real bottleneck (Section 7), and fix it. This is a skill you need to _practice_, not just read about.
6. **Try a video transformer instead of a 3D CNN** — repeat project 2 or 4 with a pretrained video transformer model and directly compare accuracy and speed trade-offs.
7. **Multi-GPU training** — scale project 6 across more than one GPU using DDP, even with just two GPUs, to build real hands-on experience before you need it at a bigger scale.
8. **A deployment-focused final project** — package a trained phase-recognition model to run close to real-time (using a streaming-friendly design or sliding window), including shrinking it down (quantization) and measuring the speed/accuracy trade-off. This is the project that best shows off your "biomedical + deployment" angle.

**Helpful links** (check these are still current when you get there, since things move fast):

- PyTorch's official video tools: https://docs.pytorch.org/vision/stable/models.html#video-classification
- PyTorchVideo (a video-focused library with ready-made models): https://pytorchvideo.org/
- Decord (the video-loading tool): https://github.com/dmlc/decord
- SlowFast paper (2019): https://arxiv.org/abs/1812.03982
- ViViT paper (2021): https://arxiv.org/abs/2103.15691
- Video Swin Transformer paper (2022): https://arxiv.org/abs/2106.13230
- NVIDIA DALI docs: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
- PyTorch Profiler guide: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Kinetics dataset info: https://www.deepmind.com/open-source/kinetics
