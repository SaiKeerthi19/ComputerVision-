import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d  # TODO: use torch.nn.functional
import os

# region DEBUG
DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(args, kwargs)


if __name__ == "__main__":
    DEBUG = os.environ.get("PYTHON_DEBUG_MODE")
    if DEBUG is not None and DEBUG.lower() == "true":
        DEBUG = True
        print("DEBUG mode is enabled")
# endregion

MAX_DISPARITY = 30      # n in the handout
KERNEL_SIZE = 5         # 5x5 box filter for smoothing
PIXEL_ROW, PIXEL_COL = 152, 152


def load_image_in_grayscale(filepath) -> torch.tensor:
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)


def sum_of_abs_diff(nparray1: np.array, nparray2: np.array) -> int:
    return (np.abs(nparray1.astype(np.int16) - nparray2.astype(np.int16))).sum().item()


def scanlines(tb_left: np.array, tb_right: np.array):
    row_idx = 152
    col_idx1 = 102
    col_len = 100
    tb_left_cropped = tb_left[row_idx][col_idx1 : col_idx1 + col_len]

    g_best = None
    d_best = None
    for d in range(col_len + 1):  # TODO: check max disparity
        tb_right_cropped = tb_right[row_idx][col_idx1 - d : col_idx1 - d + col_len]
        g = sum_of_abs_diff(tb_left_cropped, tb_right_cropped)
        if g_best == None or g < g_best:
            g_best, d_best = g, d

    return d_best


def plot_1d_array(array, title, xlabel=None, ylabel=None, save_image=True):
    domain = range(len(array))
    plt.plot(domain, array, marker="o")
    plt.xlabel(title)
    plt.ylabel(xlabel)
    plt.title(ylabel)
    plt.grid(True)
    if save_image:
        os.makedirs("figure", exist_ok=True)
        plt.savefig(f"figure/{title}.png")
    plt.show()


def plot_2d_array_as_image(array2d: np.array, title, save_image=True):
    plt.imshow(array2d, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_image:
        os.makedirs("figure", exist_ok=True)
        plt.savefig(f"figure/{title}.png")
    plt.show()


def shift_array(nparray: np.array, d: int) -> np.array:
    shifted = np.zeros_like(nparray)
    if d == 0:
        shifted[:, :] = nparray[:, :]
    elif d > 0:
        shifted[:, d:] = nparray[:, :-d]
    elif d < 0:
        shifted[:, : nparray.shape[1] + d] = nparray[:, -d:]
    return shifted


if DEBUG:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (shift_array(a, 1) == [[0, 1, 2], [0, 4, 5], [0, 7, 8]]).all()
    assert (shift_array(a, 2) == [[0, 0, 1], [0, 0, 4], [0, 0, 7]]).all()


def auto_correlation(tb_right):
    max_d = 30
    auto_correlations = []
    for d in range(max_d + 1):
        abs_diff_image = np.abs(tb_right.astype(np.int16) - shift_array(tb_right, d).astype(np.int16))
        auto_correlations.append(abs_diff_image[152][152])

    if DEBUG:
        plot_1d_array(auto_correlations, auto_correlation.__name__)
    return auto_correlations


# TODO
def convolve2d_torch(array: np.array, kernel_size: int):
    as_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(np.ones((kernel_size, kernel_size)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    convolved = nn.functional.conv2d(as_tensor, kernel, padding=kernel_size // 2)
    if DEBUG:
        assert convolved.shape == as_tensor.shape

    return np.array(convolved.squeeze().squeeze())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _abs_diff_stack(fixed, moving, sign=1, max_d=MAX_DISPARITY):
    """One absolute-difference image per disparity d = 0..max_d.
    The moving image is shifted by sign*d (right for +1, left for -1) and compared with the fixed image.
    Result shape: (max_d + 1, H, W)."""
    fixed = fixed.astype(np.int16)
    return np.stack([np.abs(fixed - shift_array(moving, sign * d).astype(np.int16))
                     for d in range(max_d + 1)])


def _smooth_stack(stack, kernel_size=KERNEL_SIZE):
    """5x5 box filter (all ones, no normalization) on every difference image."""
    return np.stack([convolve2d_torch(img, kernel_size) for img in stack])


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------
def smoothing(tb_right):
    stack = _abs_diff_stack(tb_right, tb_right, sign=1)          # same images as auto_correlation
    smoothed = _smooth_stack(stack)
    smoothed_auto_correlations = [float(v) for v in smoothed[:, PIXEL_ROW, PIXEL_COL]]

    if DEBUG:
        plot_1d_array(smoothed_auto_correlations, smoothing.__name__)
    return smoothed_auto_correlations


# ---------------------------------------------------------------------------
# Q5
# ---------------------------------------------------------------------------
def cross_correlation(tb_left, tb_right):
    stack = _abs_diff_stack(tb_left, tb_right, sign=1)           # shifted right images vs the left image
    smoothed = _smooth_stack(stack)
    cross_correlations = [float(v) for v in smoothed[:, PIXEL_ROW, PIXEL_COL]]   # smoothed values

    if DEBUG:
        plot_1d_array(cross_correlations, cross_correlation.__name__)
    return cross_correlations


# ---------------------------------------------------------------------------
# Q6
# ---------------------------------------------------------------------------
def disparity_map(tb_left, tb_right, plot_result=False):
    smoothed = _smooth_stack(_abs_diff_stack(tb_left, tb_right, sign=1))
    disparity = np.argmin(smoothed, axis=0).astype(np.uint8)     # best d for every pixel

    if plot_result:
        plot_2d_array_as_image(disparity, disparity_map.__name__)
    return disparity


# ---------------------------------------------------------------------------
# Q7
# ---------------------------------------------------------------------------
def right_left_disparity(tb_left, tb_right, plot_result=False):
    # shift the LEFT image to the left by d and compare with the right image
    smoothed = _smooth_stack(_abs_diff_stack(tb_right, tb_left, sign=-1))
    # search over magnitudes 0..30, but return NEGATIVE disparities (-30..0):
    # the match in the left image is to the right, the opposite direction of the left-right map
    disparity = -np.argmin(smoothed, axis=0).astype(np.int32)

    if plot_result:
        plot_2d_array_as_image(disparity, right_left_disparity.__name__)
    return disparity


# ---------------------------------------------------------------------------
# Q8
# ---------------------------------------------------------------------------
def disparity_check(tb_left, tb_right):
    d_left = disparity_map(tb_left, tb_right).astype(np.int32)
    d_right = right_left_disparity(tb_left, tb_right).astype(np.int32)
    H, W = d_left.shape

    cols = np.arange(W)[None, :].repeat(H, axis=0)
    rows = np.arange(H)[:, None].repeat(W, axis=1)
    x_right = cols - d_left                                      # (x, y) in left -> (x - d, y) in right
    inside = x_right >= 0

    # keep the disparity only where dR(x - d, y) = -d
    agree = np.zeros((H, W), dtype=bool)
    agree[inside] = d_right[rows[inside], x_right[inside]] == -d_left[inside]

    cleaned = np.where(agree, d_left, 0).astype(np.uint8)

    if DEBUG:
        plot_2d_array_as_image(cleaned, disparity_check.__name__)
    return cleaned


# ---------------------------------------------------------------------------
# Q9
# ---------------------------------------------------------------------------
def reconstruction(tb_left, tb_right):
    # Fill your code hear
    disparity = disparity_check(tb_left, tb_right)

    color = cv.imread("tsukuba_left.png", cv.IMREAD_COLOR)
    color = cv.cvtColor(color, cv.COLOR_BGR2RGB)                 # colors come from the left image

    focal_length = 100.0    # nominal values (pixels / arbitrary units), picked so the depth looks right
    baseline = 20.0

    rows, cols = np.nonzero(disparity)                           # only pixels that passed the check
    d = disparity[rows, cols].astype(np.float32)

    # orthographic projection: x, y are just the pixel position, z comes from the disparity
    x = cols.astype(np.float32)
    y = -rows.astype(np.float32)                                 # flip so the image is not upside down
    z = -focal_length * baseline / d                             # depth = f * b / d (minus: far points go back)
    rgb = color[rows, cols]

    with open("kermit.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(x)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xi, yi, zi, (r, g, b) in zip(x, y, z, rgb):
            f.write(f"{xi:.3f} {yi:.3f} {zi:.3f} {r} {g} {b}\n")

    return len(x)


if __name__ == "__main__":
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")
    # scanlines(tb_left, tb_right)
    # auto_correlation(tb_right)
    # smoothing(tb_right)
    # cross_correlation(tb_left, tb_right)
    # disparity_map(tb_left, tb_right, plot_result=True)
    # right_left_disparity(tb_left, tb_right, plot_result=True)
    disparity_check(tb_left, tb_right)
    reconstruction(tb_left, tb_right)
