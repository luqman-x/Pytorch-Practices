# PyTorch Loss Functions: Focal Loss, Dice Loss, and Combined Losses

A practical guide to understanding and implementing **Focal Loss**,
**Dice Loss**, and **combined losses** in PyTorch, with a focus on
medical imaging and segmentation.

------------------------------------------------------------------------

## 1. What is a loss function?

A loss function tells a neural network:

> **How wrong is my prediction?**

For example, in tumor segmentation:

``` text
MRI image
   ↓
Neural Network
   ↓
Prediction
   ↓
Loss
   ↓
Backpropagation
   ↓
Update model weights
```

For binary segmentation, the model might output tumor probabilities such
as:

``` text
0.01  0.03  0.02  0.80
0.05  0.90  0.95  0.70
0.01  0.85  0.92  0.10
```

The ground truth might be:

``` text
0  0  0  1
0  1  1  1
0  1  1  0
```

The loss measures the difference between the prediction and target.

------------------------------------------------------------------------

# 2. Why ordinary BCE can struggle with segmentation

For binary classification or segmentation, a common starting point is:

``` python
loss_fn = torch.nn.BCEWithLogitsLoss()
```

This is an excellent baseline.

However, imagine a medical image where:

``` text
Background: 99%
Tumor:       1%
```

A model could predict almost everything as background and still achieve
around:

``` text
99% pixel accuracy
```

That is not useful if the real goal is to detect the tumor.

This is where losses such as **Focal Loss** and **Dice Loss** become
useful.

------------------------------------------------------------------------

# 3. Focal Loss

The main idea behind Focal Loss is:

> **Don't let easy examples dominate the training. Focus more on
> difficult examples.**

Imagine these predictions for a positive target:

``` text
Prediction A = 0.99
Prediction B = 0.80
Prediction C = 0.55
Prediction D = 0.10
```

Prediction A is easy and correct.

Prediction D is very wrong.

Focal Loss reduces the contribution of easy examples and gives difficult
examples more influence.

------------------------------------------------------------------------

# 4. Focal Loss equation

For binary classification, a common form is:

\[ FL(p_t)=-`\alpha`{=tex}\_t(1-p_t)\^`\gamma`{=tex}`\log`{=tex}(p_t) \]

The important components are:

-   (p_t): probability assigned to the correct class
-   (`\gamma`{=tex}): focusing parameter
-   (`\alpha`{=tex}\_t): class-balancing factor

The important term is:

\[ (1-p_t)\^`\gamma`{=tex} \]

If the model is very confident and correct:

``` text
p_t = 0.99
```

then:

``` text
1 - p_t = 0.01
```

so the example's loss is strongly reduced.

If:

``` text
p_t = 0.20
```

then:

``` text
1 - p_t = 0.80
```

so the example receives much more attention.

------------------------------------------------------------------------

# 5. What is gamma?

The main Focal Loss parameter is:

``` python
gamma
```

A common starting value is:

``` python
gamma = 2.0
```

Think of gamma as:

> **How aggressively should I reduce the influence of easy examples?**

Typical intuition:

  Gamma   Behavior
  ------- -----------------------------------------------
  0       Similar to ordinary BCE
  1       Moderate focusing
  2       Common starting point
  5+      Strong focusing; may make optimization harder

For most experiments, start with:

``` python
gamma = 2.0
```

------------------------------------------------------------------------

# 6. What is alpha?

You may also see:

``` python
alpha
```

Alpha handles **class imbalance**.

For example:

``` text
Background = 95%
Tumor      = 5%
```

You may want tumor pixels to have greater importance.

Therefore:

``` text
gamma → focus on hard examples

alpha → balance class importance
```

------------------------------------------------------------------------

# 7. Modern PyTorch implementation of binary Focal Loss

For binary classification or segmentation, build Focal Loss around
**logits**, not probabilities.

Your model should output:

``` python
logits = model(x)
```

Use `binary_cross_entropy_with_logits`, which is numerically stable.

``` python
import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:

    targets = targets.float()

    # Stable BCE loss for each element
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    # Probability of the correct class
    p_t = torch.exp(-bce)

    loss = (1 - p_t).pow(gamma) * bce

    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()
```

This implementation avoids manually doing:

``` python
sigmoid()
log()
```

for the BCE portion.

------------------------------------------------------------------------

# 8. Practice Focal Loss

``` python
import torch

logits = torch.tensor([
    5.0,    # very confident positive
    2.0,    # reasonably confident positive
    0.0,    # uncertain
    -2.0,   # wrong for positive target
])

targets = torch.tensor([
    1,
    1,
    1,
    1,
])

loss = focal_loss(
    logits,
    targets,
    gamma=2.0,
)

print(loss)
```

Interpretation:

``` text
logit = 5
    ↓
probability ≈ 0.993
    ↓
easy example
    ↓
strongly down-weighted

logit = -2
    ↓
probability ≈ 0.119
    ↓
hard/wrong example
    ↓
receives much more attention
```

------------------------------------------------------------------------

# 9. Why use logits?

A good PyTorch habit is:

``` python
logits = model(x)

loss = focal_loss(logits, y)
```

rather than putting sigmoid before the loss:

``` python
probabilities = torch.sigmoid(model(x))
loss = F.binary_cross_entropy(probabilities, y)
```

The logits-based approach is more numerically stable because
`binary_cross_entropy_with_logits` combines the sigmoid and BCE
calculation safely.

------------------------------------------------------------------------

# 10. Dice Loss

Dice Loss is extremely important for **image segmentation**.

The Dice coefficient measures overlap between:

``` text
Prediction
```

and:

``` text
Ground truth
```

The formula is:

\[ Dice = `\frac{2|P \cap G|}{|P|+|G|}`{=tex} \]

where:

-   \(P\) = predicted region
-   \(G\) = ground-truth region

Dice is approximately between:

``` text
0 → terrible overlap

1 → perfect overlap
```

So Dice Loss is:

\[ DiceLoss = 1-Dice \]

------------------------------------------------------------------------

# 11. Simple Dice example

Suppose:

``` text
Ground truth:

1 1 0 0
1 1 0 0
0 0 0 0
```

Prediction:

``` text
1 1 0 0
1 0 0 0
0 0 0 0
```

Then:

``` text
Ground truth tumor pixels = 4
Predicted tumor pixels     = 3
Overlap                    = 3
```

Therefore:

\[ Dice = `\frac{2(3)}{4+3}`{=tex} \]

\[ Dice `\approx 0.857`{=tex} \]

The prediction has good overlap.

------------------------------------------------------------------------

# 12. Why Dice Loss is useful for medical segmentation

Consider a 512 × 512 image:

``` text
512 × 512 = 262,144 pixels
```

Maybe the tumor occupies only:

``` text
2,000 pixels
```

Most pixels are background.

Dice focuses directly on:

> **How well does my predicted region overlap the actual region?**

This makes it particularly useful for:

-   Tumor segmentation
-   Organ segmentation
-   Lesion segmentation
-   Cell segmentation
-   Blood vessel segmentation
-   Wound segmentation

------------------------------------------------------------------------

# 13. Modern binary Dice Loss implementation

``` python
import torch


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:

    targets = targets.float()

    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probs.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return 1.0 - dice.mean()
```

Notice:

``` python
probs = torch.sigmoid(logits)
```

Here sigmoid is required because Dice operates on probabilities.

------------------------------------------------------------------------

# 14. Why flatten?

A binary segmentation model might output:

``` text
[B, 1, H, W]
```

For example:

``` text
[8, 1, 256, 256]
```

This means:

``` text
8 images
1 channel
256 × 256 pixels
```

We want to calculate Dice for each image.

This:

``` python
probs.flatten(1)
```

turns:

``` text
[8, 1, 256, 256]
```

into:

``` text
[8, 65536]
```

Each image becomes one long vector of pixels.

------------------------------------------------------------------------

# 15. Why use smooth?

Dice contains a division operation:

\[ Dice = `\frac{2intersection + smooth}`{=tex} {prediction + target +
smooth} \]

The `smooth` value helps avoid problematic division by zero.

For example, if both prediction and target are completely empty:

``` text
prediction = 0
target = 0
```

Without smoothing:

``` text
0 / 0
```

A common starting value is:

``` python
smooth = 1.0
```

------------------------------------------------------------------------

# 16. Practice Dice Loss

``` python
logits = torch.tensor([
    [
        [
            [5.0, 5.0, -5.0],
            [5.0, 5.0, -5.0],
            [-5.0, -5.0, -5.0],
        ]
    ]
])

targets = torch.tensor([
    [
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    ]
])

loss = dice_loss(logits, targets)

print(loss)
```

Because the logits correspond closely to the target mask, Dice Loss
should be close to zero.

------------------------------------------------------------------------

# 17. BCE vs Focal vs Dice

Think about the losses this way:

### BCE

> **"Is each pixel classified correctly?"**

### Focal Loss

> **"Focus more on the pixels/examples that are difficult."**

### Dice Loss

> **"Does my predicted region overlap the target region?"**

This is why Dice is particularly attractive for segmentation.

------------------------------------------------------------------------

# 18. Why combine losses?

The losses capture different properties.

Imagine:

``` text
Ground truth:

████
████
░░░░
░░░░
```

Prediction:

``` text
████
██░░
░░░░
░░░░
```

Dice understands:

> "The predicted tumor region overlaps the real tumor region reasonably
> well."

BCE or Focal Loss gives you pixel-level classification information.

So we can combine them:

\[ Loss = `\lambda`{=tex}\_1 BCE + `\lambda`{=tex}\_2 DiceLoss \]

or:

\[ Loss = `\lambda`{=tex}\_1 FocalLoss + `\lambda`{=tex}\_2 DiceLoss \]

Combined losses are common in medical segmentation.

------------------------------------------------------------------------

# 19. BCE + Dice Loss

A practical starting point is:

``` text
BCE + Dice
```

Implementation:

``` python
import torch.nn.functional as F


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
) -> torch.Tensor:

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
    )

    dice = dice_loss(
        logits,
        targets,
    )

    return bce + dice_weight * dice
```

Usage:

``` python
loss = bce_dice_loss(logits, targets)
```

------------------------------------------------------------------------

# 20. Focal + Dice

For highly imbalanced segmentation such as:

``` text
Background → 99%
Lesion     → 1%
```

you might try:

``` python
def focal_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    focal_weight: float = 1.0,
    dice_weight: float = 1.0,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:

    focal = focal_loss(
        logits,
        targets,
        alpha=alpha,
        gamma=gamma,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    return (
        focal_weight * focal
        + dice_weight * dice
    )
```

Usage:

``` python
loss = focal_dice_loss(
    logits,
    targets,
    focal_weight=1.0,
    dice_weight=1.0,
    gamma=2.0,
)
```

------------------------------------------------------------------------

# 21. What does the weighting mean?

Suppose:

``` python
loss = 0.7 * focal + 1.0 * dice
```

You are assigning a smaller coefficient to Focal Loss than Dice Loss.

You can experiment with:

``` python
focal_weight = 1.0
dice_weight = 1.0
```

then:

``` python
focal_weight = 0.5
dice_weight = 1.0
```

or:

``` python
focal_weight = 1.0
dice_weight = 2.0
```

The individual loss scales matter, so these weights should be tuned
experimentally rather than treated as fixed percentages.

------------------------------------------------------------------------

# 22. Complete segmentation training example

Imagine:

``` text
Input:
MRI → [B, 1, H, W]

Target:
Mask → [B, 1, H, W]

Model:
U-Net
```

Training:

``` python
model = UNet()

for images, masks in train_loader:

    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()

    logits = model(images)

    loss = focal_dice_loss(
        logits,
        masks,
        focal_weight=1.0,
        dice_weight=1.0,
        gamma=2.0,
    )

    loss.backward()

    optimizer.step()
```

This is the essential training workflow.

------------------------------------------------------------------------

# 23. Do not put sigmoid inside the model

For binary segmentation, a clean model architecture is:

``` python
class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # encoder...
        # decoder...

        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        ...
        return self.output(x)
```

The output is:

``` text
logits
```

not probabilities.

Then:

``` python
loss = focal_dice_loss(logits, masks)
```

Internally:

``` text
Focal:
logits → BCEWithLogits

Dice:
logits → sigmoid → probabilities
```

This separation is clean and avoids applying sigmoid twice.

------------------------------------------------------------------------

# 24. Prediction time is different

During inference:

``` python
model.eval()

with torch.inference_mode():

    logits = model(image)

    probabilities = torch.sigmoid(logits)

    mask = probabilities > 0.5
```

So:

### Training

``` text
image
 ↓
model
 ↓
logits
 ↓
loss
```

### Inference

``` text
image
 ↓
model
 ↓
logits
 ↓
sigmoid
 ↓
probability
 ↓
threshold
 ↓
binary mask
```

------------------------------------------------------------------------

# 25. Dice Loss for multiclass segmentation

There are different segmentation problems.

### Binary segmentation

``` text
0 = background
1 = tumor
```

Output:

``` python
[B, 1, H, W]
```

### Multiclass segmentation

``` text
0 = background
1 = liver
2 = kidney
3 = tumor
```

Output:

``` python
[B, 4, H, W]
```

For multiclass segmentation, use a multiclass Dice implementation.

------------------------------------------------------------------------

# 26. Modern multiclass Dice Loss

``` python
import torch
import torch.nn.functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:

    num_classes = logits.shape[1]

    probs = F.softmax(logits, dim=1)

    targets_one_hot = F.one_hot(
        targets.long(),
        num_classes=num_classes,
    )

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    probs = probs.flatten(2)
    targets_one_hot = targets_one_hot.flatten(2)

    intersection = (
        probs * targets_one_hot
    ).sum(dim=2)

    dice = (
        2.0 * intersection + smooth
    ) / (
        probs.sum(dim=2)
        + targets_one_hot.sum(dim=2)
        + smooth
    )

    return 1.0 - dice.mean()
```

Usage:

``` python
logits = model(images)

loss = multiclass_dice_loss(
    logits,
    masks,
)
```

------------------------------------------------------------------------

# 27. Multiclass CrossEntropy + Dice

For multiclass medical segmentation, a sensible starting point is:

``` text
CrossEntropy + Dice
```

Implementation:

``` python
def ce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:

    ce = F.cross_entropy(
        logits,
        targets.long(),
    )

    dice = multiclass_dice_loss(
        logits,
        targets,
    )

    return (
        ce_weight * ce
        + dice_weight * dice
    )
```

Then:

``` python
loss = ce_dice_loss(
    logits,
    masks,
)
```

------------------------------------------------------------------------

# 28. Which combined loss should you start with?

  Problem                                 Starting loss
  --------------------------------------- ------------------------------------
  Balanced binary classification          BCEWithLogitsLoss
  Imbalanced binary classification        Focal Loss
  Binary segmentation                     BCE + Dice
  Highly imbalanced binary segmentation   Focal + Dice
  Multiclass segmentation                 CrossEntropy + Dice
  Multiclass, severe imbalance            Weighted CE + Dice or Focal + Dice

Don't automatically assume that a more complicated loss is better.

Start with a simple baseline, then add complexity.

------------------------------------------------------------------------

# 29. Focal Loss is not automatically better than BCE

For example:

``` text
BCE:
Excellent baseline

Focal:
May improve minority/hard-example performance
```

But if your dataset isn't heavily imbalanced, Focal Loss can sometimes
make optimization harder.

A useful experiment is:

``` text
Experiment 1:
BCE

Experiment 2:
Focal

Experiment 3:
BCE + Dice

Experiment 4:
Focal + Dice
```

Then compare:

``` text
Validation Dice
Precision
Recall
F1
IoU
Sensitivity
Specificity
```

Do not compare only the training loss.

------------------------------------------------------------------------

# 30. Biomedical example

Suppose you're detecting tumors from MRI.

Dataset:

``` text
10,000 images

Tumor:
1,000

No tumor:
9,000
```

For classification:

``` python
loss_fn = focal_loss
```

For segmentation:

``` python
loss_fn = focal_dice_loss
```

Why?

### Classification

``` text
"Does this image contain a tumor?"
```

Focal Loss helps focus on difficult examples.

### Segmentation

``` text
"Which pixels belong to the tumor?"
```

Dice helps optimize region overlap.

Therefore:

``` text
Tumor classification
       ↓
     Focal

Tumor segmentation
       ↓
 Focal + Dice
```

------------------------------------------------------------------------

# 31. Tversky Loss

For biomedical segmentation, you will eventually encounter **Tversky
Loss**.

It extends Dice and lets you control the relative importance of:

``` text
False positives
False negatives
```

This is useful when:

> Missing a disease region is much worse than producing some false
> positives.

For example, in lesion detection, you may prioritize
**sensitivity/recall**.

A useful progression is:

``` text
BCE
 ↓
Focal Loss
 ↓
Dice Loss
 ↓
Focal + Dice
 ↓
Tversky Loss
 ↓
Focal Tversky Loss
```

------------------------------------------------------------------------

# 32. Important practical rules

## Rule 1 --- Binary segmentation outputs logits

Your model should generally output:

``` python
[B, 1, H, W]
```

with raw logits.

------------------------------------------------------------------------

## Rule 2 --- Use BCEWithLogitsLoss

Prefer:

``` python
F.binary_cross_entropy_with_logits(
    logits,
    targets,
)
```

rather than manually doing:

``` python
probs = torch.sigmoid(logits)
F.binary_cross_entropy(probs, targets)
```

for the BCE component.

------------------------------------------------------------------------

## Rule 3 --- Dice needs probabilities

For binary Dice:

``` python
probs = torch.sigmoid(logits)
```

------------------------------------------------------------------------

## Rule 4 --- Multiclass Dice uses softmax

For multiclass segmentation:

``` python
probs = F.softmax(logits, dim=1)
```

------------------------------------------------------------------------

## Rule 5 --- Don't use sigmoid and softmax for the same multiclass output

Binary and multiclass segmentation use different output formulations.

------------------------------------------------------------------------

## Rule 6 --- Evaluate with appropriate metrics

For medical segmentation, evaluate:

``` text
Dice
IoU
Sensitivity
Specificity
Precision
Recall
```

rather than relying only on the training loss.

------------------------------------------------------------------------

# 33. Practice project

A useful hands-on exercise is to create a synthetic segmentation dataset
where:

``` text
Background = 95%
Object = 5%
```

Train the same small CNN three times:

``` text
Experiment A
BCE

Experiment B
Dice

Experiment C
BCE + Dice
```

Then record:

``` text
Training loss
Validation loss
Dice score
IoU
Precision
Recall
```

After that, add:

``` text
Experiment D
Focal

Experiment E
Focal + Dice
```

This makes the practical differences between the losses much easier to
understand.

------------------------------------------------------------------------

# 34. Recommended learning progression

A good order for learning these concepts is:

``` text
1. BCEWithLogitsLoss
       ↓
2. Focal Loss
       ↓
3. Dice coefficient
       ↓
4. Dice Loss
       ↓
5. BCE + Dice
       ↓
6. Multiclass Dice
       ↓
7. CrossEntropy + Dice
       ↓
8. Tversky Loss
       ↓
9. Focal Tversky Loss
```

The most important mental model is:

``` text
BCE
→ Pixel-level classification

Focal
→ Focus on difficult / imbalanced examples

Dice
→ Region overlap

Combined losses
→ Optimize multiple useful properties simultaneously
```

For biomedical image segmentation, this foundation will prepare you well
for more advanced segmentation losses and architectures such as U-Net,
U-Net++, Attention U-Net, DeepLab, and transformer-based segmentation
models.
