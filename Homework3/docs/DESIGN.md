# Assignment 3 – Design notes

## Conventions

| Item | Convention |
|------|------------|
| Input | OpenCV `imread` → BGR, `uint8`, shape (H, W, 3) |
| Colour order | Swapped to **RGB** first in every colour method |
| Tensors | `torch.from_numpy(...)` then cast with `.to(dtype)` |

## Q1 – Tensor arithmetic

* **1a** `float32` keeps values 0–255 but allows decimals and values above 255.
* **1b** `+ 100.0` on a float tensor never wraps (max 355).
* **1c** `uint8` arithmetic wraps modulo 256 (200 + 100 → 44). Casting to `int16` first,
  adding, then `clamp(0, 255)` and casting back gives true saturation (→ 255).

## Q2 – Gaussian noise

`torch.randn` gives mean 0 / std 1, so `* 100.0` gives σ = 100 gray levels.
Result is divided by 255 and clamped to [0, 1] because float images are expected in that range.

## Q3 – Normalization

`x.mean(dim=(0, 1))` and `x.std(dim=(0, 1))` give one value per channel (shape `[3]`);
broadcasting applies them to every pixel.

* **3a** own statistics, no clamp → mean 0, std 1 per channel.
* **3b** scale to [0, 1], use ImageNet mean `[0.485, 0.456, 0.406]` and std
  `[0.229, 0.224, 0.225]`, then clamp to [0, 1].

## Q4 – Dimension rearrange

`permute(2, 0, 1)` turns (H, W, C) into (C, H, W); `unsqueeze(0)` adds the batch axis → (1, C, H, W),
the layout PyTorch CNNs expect.

## Q5 – Stride convolution

* Scharr_x = `[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]` (horizontal gradient → vertical edges).
* `torch.nn.functional.conv2d` computes **cross-correlation**; true convolution flips the kernel,
  so the kernel is flipped with `torch.flip(k, dims=[0, 1])` first (otherwise every value has the
  wrong sign).
* `padding=1` keeps the size for stride 1; `stride=2` halves each dimension.
