"""
Demo runner for Assignment3.py - prints every result and saves images to results/.

Usage (from this folder):
    python run_demo.py
"""

import os

import cv2 as cv
import numpy as np
import torch

from Assignment3 import Assignment3

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
OUT_DIR = os.path.join(HERE, "results")


def to_uint8_rgb(t):
    """float tensor in 0..1 (H, W, 3) -> uint8 BGR array for cv.imwrite."""
    a = (t.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return cv.cvtColor(a, cv.COLOR_RGB2BGR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv.imread(os.path.join(IMG_DIR, "original_image.png"))
    img_cat = cv.imread(os.path.join(IMG_DIR, "cat_eye.jpg"), cv.IMREAD_GRAYSCALE)
    a = Assignment3()

    torch_img = a.torch_image_conversion(img)
    bright = a.brighten(torch_img)
    sat = a.saturation_arithmetic(img)
    noisy = a.add_noise(torch_img)
    norm = a.normalization_image(img)
    inet = a.Imagenet_norm(img)
    rearr = a.dimension_rearrange(img)
    stride = a.stride(img_cat)

    print("Q1a:", tuple(img.shape), img.dtype, "->", torch_img.shape, torch_img.dtype)
    print("Q1b:", bright.shape, bright.dtype, "max", bright.max().item())
    print("Q1c:", sat.dtype, "min", sat.min().item(), "max", sat.max().item())
    print("Q2 :", noisy.dtype, "range", noisy.min().item(), noisy.max().item())
    print("Q3a: mean", norm.mean(dim=(0, 1)), "std", norm.std(dim=(0, 1)))
    print("Q3b:", inet.dtype, "range", inet.min().item(), inet.max().item())
    print("Q4 :", rearr.shape, rearr.dtype)
    print("Q5 :", stride.shape, stride.type())

    cv.imwrite(os.path.join(OUT_DIR, "1a_tensor.png"), to_uint8_rgb(torch_img / 255))
    cv.imwrite(os.path.join(OUT_DIR, "1b_brighten.png"), to_uint8_rgb(bright / 255))
    cv.imwrite(os.path.join(OUT_DIR, "1c_saturation.png"), cv.cvtColor(sat.numpy(), cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(OUT_DIR, "2_noise.png"), to_uint8_rgb(noisy))
    cv.imwrite(os.path.join(OUT_DIR, "3a_own_norm.png"),
               to_uint8_rgb(((norm - norm.min()) / (norm.max() - norm.min())).float()))
    cv.imwrite(os.path.join(OUT_DIR, "3b_imagenet_norm.png"), to_uint8_rgb(inet.float()))
    s = stride.abs()
    cv.imwrite(os.path.join(OUT_DIR, "5_stride.png"),
               (torch.clamp(s / s.max() * 255 * 3, 0, 255)).numpy().astype(np.uint8))
    print("Saved images to", OUT_DIR)


if __name__ == "__main__":
    main()
