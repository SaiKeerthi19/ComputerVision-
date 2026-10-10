# Design notes and maths

This file explains *why* each task is implemented the way it is.

## Conventions

| Item | Convention |
|------|------------|
| Array layout | `image[row, col, channel]` → `image[y, x, c]` |
| Channel order | OpenCV BGR: 0 = Blue, 1 = Green, 2 = Red |
| Pixel type | `uint8`, range 0–255 |
| Axes | origin top-left, x → right, y → **down** |
| Angles | because y points down, the standard rotation matrix turns things **clockwise** on screen |

## Task 1 – Analyse

`type(img)` is `numpy.ndarray` (the container); `img.dtype` is `uint8` (each value 0–255);
`img.shape` is `(height, width, channels)`.

## Task 2 – Red image

Zero channels 0 (B) and 1 (G) on a copy. Channel 2 (R) is untouched.

## Task 3 – Negative

`255 − v` maps 0 ↔ 255 for every channel. NumPy applies it to all values at once and the
result stays in 0–255.

## Task 4 – Swap R and B

Fancy indexing `img[:, :, [2, 1, 0]]` reorders the channel axis: `[B, G, R] → [R, G, B]`.

## Task 5 – Foliage detection

Boolean mask `(G >= 50) & (R < 50) & (B < 50)`; `&` is element-wise AND (brackets are
required because of operator precedence). Output is a single channel: 255 where the mask is
True, 0 elsewhere. The rule is strict, so only deep-green pixels pass.

## Task 6 – Translation

Affine matrix `M = [[1, 0, tx], [0, 1, ty]]`, i.e. `x' = x + tx`, `y' = y + ty`.
`cv2.warpAffine(img, M, (w, h))` keeps the size and fills uncovered pixels with 0
(`BORDER_CONSTANT`, value 0). Note OpenCV takes size as `(width, height)`.

Equivalent manual version:

```python
out = np.zeros_like(img)
out[ty:, tx:] = img[:h - ty, :w - tx]
```

## Task 7 – Rotate 90° clockwise

Mapping: `new_row = old_col`, `new_col = (h − 1) − old_row`; output shape `(w, h, 3)`.
Equivalent forms: `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)`, `np.rot90(img, k=-1)`,
or transpose then flip left-right.

## Task 8 – Similarity transform (inverse mapping)

Forward model, applied in the order scale → rotate → translate:

```
q = R (s p) + t,      R = [[cos θ, −sin θ],
                           [sin θ,  cos θ]]
```

**Why inverse mapping:** pushing input pixels forward leaves holes when the image is
enlarged. Instead, for **every output pixel** `q` compute its source:

```
p = Rᵀ (q − t) / s
```

then round `p` to the nearest integer pixel (nearest neighbour). Sources outside the input
stay black. Implemented with `np.mgrid` so all pixels are processed at once.

Check: output pixel `(100, 100)` with `t = (100, 100)` maps to input `(0, 0)`, so the origin
moved to `(100, 100)`.

## Task 9 – Grayscale

`gray = round((3R + 6G + B) / 10)`. Channels are cast to `float64` first because
`uint8` arithmetic would overflow (6 × 200 = 1200). Green has the largest weight
because human vision is most sensitive to green.

## Task 10 – Moments

With `I(x, y) ∈ {0, 1}`:

| Moment | Formula | Meaning |
|--------|---------|---------|
| m00 | Σ I | area |
| m10, m01 | Σ x I, Σ y I | first-order sums |
| x̄, ȳ | m10/m00, m01/m00 | centroid |
| μ20 | Σ (x − x̄)² I | horizontal spread |
| μ02 | Σ (y − ȳ)² I | vertical spread |
| μ11 | Σ (x − x̄)(y − ȳ) I | diagonal correlation (tilt) |

## Task 11 – Orientation, eccentricity, ellipse

```
a = μ20/m00,  b = μ11/m00,  c = μ02/m00
λ1 ≥ λ2 = eigenvalues of [[a, b], [b, c]]
orientation  = degrees(0.5 · arctan(2 μ11 / (μ20 − μ02)))
eccentricity = sqrt((λ1 − λ2) / λ1)
full axis    = 4 √λ   → cv2.ellipse half-axis = int(full_axis // 2)
```

`np.linalg.eigvalsh` returns eigenvalues in ascending order, so `λ2 = lam[0]`,
`λ1 = lam[1]`. The ellipse is drawn with `cv2.ellipse(img, (int(x̄), int(ȳ)), axes,
orientation, 0, 360, (0, 0, 255), 1)` on a copy of the BGR image, so it appears on top of
the white outline.
