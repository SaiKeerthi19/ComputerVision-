"""
Demo runner for Assignment1.py

Runs all 11 tasks, prints the numeric results and saves every output image
into the ``results/`` folder, plus one overview picture
(``results/results_overview.png``).

Usage:
    python run_demo.py
"""

import os

import cv2
import numpy as np

from Assignment1 import ComputerVisionAssignment

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")
OUT_DIR = os.path.join(HERE, "results")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- Tasks 1-9 on the colour photo ----------
    cva = ComputerVisionAssignment(
        os.path.join(IMG_DIR, "original_image.png"),
        os.path.join(IMG_DIR, "binary_image.png"),
    )
    if cva.image is None:
        raise FileNotFoundError("images/original_image.png not found")

    image_type, pixel_type, shape = cva.load_and_analyze_image()
    print("Task 1")
    print(f"  Image data type : {image_type}")
    print(f"  Pixel data type : {pixel_type}  (uint8: {pixel_type == np.uint8})")
    print(f"  Image dimensions: {shape}")

    outputs = {
        "2_red": cva.create_red_image(),
        "3_negative": cva.create_photographic_negative(),
        "4_swapped": cva.swap_color_channels(),
        "5_foliage": cva.foliage_detection(),
        "6_shifted": cva.shift_image(),
        "7_rotated": cva.rotate_image(),
        "8_similarity": cva.similarity_transform(scale=2.0, theta=45.0, shift=[100, 100]),
        "9_gray": cva.convert_to_grayscale(),
    }

    # ---------- Tasks 10-11 on the binary glasses image ----------
    glasses_path = os.path.join(IMG_DIR, "glasses_outline.png")
    glasses = ComputerVisionAssignment(glasses_path, glasses_path)

    m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = glasses.compute_moments()
    print("\nTask 10 - Moments")
    print(f"  Raw moments     : M00 = {m00}, M10 = {m10}, M01 = {m01}")
    print(f"  Centroid        : x_bar = {x_bar:.4f}, y_bar = {y_bar:.4f}")
    print(f"  Central moments : mu20 = {mu20:.4f}, mu02 = {mu02:.4f}, mu11 = {mu11:.4f}")

    orientation, eccentricity, ellipse_img = glasses.compute_orientation_and_eccentricity()
    print("\nTask 11 - Shape")
    print(f"  Orientation     : {orientation:.4f} degrees (clockwise)")
    print(f"  Eccentricity    : {eccentricity:.4f}")
    outputs["11_ellipse"] = ellipse_img

    # ---------- Save every result ----------
    for name, img in outputs.items():
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), img)
    print(f"\nSaved {len(outputs)} images to {OUT_DIR}")

    # ---------- Optional overview picture (needs matplotlib) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [("Original", cva.image)] + list(outputs.items())
        fig, axes = plt.subplots(2, 5, figsize=(20, 7))
        for ax, (title, img) in zip(axes.flat, panels):
            if img.ndim == 2:                       # gray / binary image
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:                                   # BGR -> RGB for display
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "results_overview.png"), dpi=90)
        print("Saved results/results_overview.png")
    except ImportError:
        print("matplotlib not installed - skipped overview picture")


if __name__ == "__main__":
    main()
