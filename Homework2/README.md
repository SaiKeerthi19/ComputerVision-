# Computer Vision – Filtering and Edge Detection (EE P 596, HW 2)

Python implementation of basic image filtering for EE P 596 Computer Vision at the University of Washington, using OpenCV, NumPy and SciPy.

## Tasks

| # | Task | Method |
|---|---|---|
| 1 | Floodfill | Stack-based, 8-connected neighbors; fills the ant's face red from a seed point |
| 2 | Gaussian smoothing | Separable kernel 0.25·[1 2 1] (vertical 3×1, then horizontal 1×3), applied 5 times in succession |
| 3 | Vertical derivative | Smooth horizontally with 0.25·[1 2 1], differentiate vertically with 0.5·[1 0 −1] (Sobel) |
| 4 | Horizontal derivative | The transpose of Task 3 |
| 5 | Gradient magnitude | Manhattan norm \|gx\| + \|gy\| from Tasks 3 and 4 |
| 6 | Built-in convolution | Task 3 repeated with `scipy.signal.convolve2d` (which flips the kernel, so light and dark are swapped) |
| 7 | Repeated box filtering | `[1 1 1]` convolved with itself; the result approaches a Gaussian (bell) curve |

Tasks 2–4 use hand-written convolution (only add and multiply in a double for-loop over the pixels), with zero padding and no kernel flip.

## Files

- `Assignment2.py` – all seven tasks in the `ComputerVisionAssignment` class
- `ant_outline.png` – input for Task 1
- `cat_eye.jpg` – input for Tasks 2–6

## Run
cat > README.md <<'EOF'
# Computer Vision (EE P 596)

Homework for EE P 596 Computer Vision: Classical and Deep Methods, University of Washington.

| Folder | Topic |
|---|---|
| [Homework1](Homework1/) | Classical image processing: color channels, transforms, binary image analysis |
| [Homework2](Homework2/) | Floodfill, Gaussian smoothing, derivatives, gradient magnitude, box filtering |

Each folder has its own README with setup and run instructions.
