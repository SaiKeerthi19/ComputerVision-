"""
EE P 596 Computer Vision: Classical and Deep Methods - Homework 1
==================================================================

Basic image processing and binary image analysis with OpenCV and NumPy.

All methods live in one class, ``ComputerVisionAssignment``. Each method
reads the images loaded in ``__init__`` and *returns* its result without
printing or saving anything, so the class can be imported and tested by an
autograder or reused in other projects.

Conventions used everywhere in this file
----------------------------------------
* Images are NumPy arrays (``numpy.ndarray``).
* Colour images use OpenCV's native layout: shape (height, width, 3),
  channels in **B, G, R** order, pixel type ``uint8`` (0..255).
* Coordinates: ``x`` = column (left -> right), ``y`` = row (top -> bottom).
  Because y points *down*, a positive rotation angle looks *clockwise*.

To see the results (printed values and saved images), run ``run_demo.py``.
"""

import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    """Holds the input images and implements Tasks 1-11 of Homework 1."""

    def __init__(self, image_path, binary_image_path):
        """
        Load the images once so every task can reuse them.

        Args:
            image_path: path to a colour image (loaded as BGR, uint8).
            binary_image_path: path to a black/white image (loaded as one
                gray channel, uint8).
        """
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # ------------------------------------------------------------------
    # Task 0 - environment check (ungraded)
    # ------------------------------------------------------------------
    def check_package_versions(self):
        """Import the main libraries. Uncomment the prints to see versions."""
        import numpy as np
        import matplotlib
        import cv2

        # print(np.__version__)
        # print(matplotlib.__version__)
        # print(cv2.__version__)

    # ------------------------------------------------------------------
    # Task 1 - image load / analyse
    # ------------------------------------------------------------------
    def load_and_analyze_image(self):
        """
        Describe the loaded colour image.

        Returns:
            (Image_data_type, Pixel_data_type, Image_shape)
            e.g. (numpy.ndarray, dtype('uint8'), (321, 433, 3))
        """
        Image_data_type = type(self.image)   # the container: numpy.ndarray
        Pixel_data_type = self.image.dtype   # the numbers inside: uint8 (0-255)
        Image_shape = self.image.shape       # (height, width, channels)

        # print(f"Image data type: {Image_data_type}")
        # print(f"Pixel data type: {Pixel_data_type}")
        # print(f"Image dimensions: {Image_shape}")

        return Image_data_type, Pixel_data_type, Image_shape

    # ------------------------------------------------------------------
    # Task 2 - keep only the red channel
    # ------------------------------------------------------------------
    def create_red_image(self):
        """
        Set the blue and green channels to zero so only red remains.

        Returns:
            BGR uint8 image of the same size that looks red.
        """
        red_image = self.image.copy()   # copy so the original stays unchanged
        red_image[:, :, 0] = 0          # channel 0 = Blue  -> off
        red_image[:, :, 1] = 0          # channel 1 = Green -> off
        return red_image                # channel 2 = Red is untouched

    # ------------------------------------------------------------------
    # Task 3 - photographic negative
    # ------------------------------------------------------------------
    def create_photographic_negative(self):
        """
        Invert every value: new = 255 - old (bright <-> dark).

        Returns:
            BGR uint8 negative image.
        """
        negative_image = (255 - self.image).astype(np.uint8)
        return negative_image

    # ------------------------------------------------------------------
    # Task 4 - swap red and blue channels
    # ------------------------------------------------------------------
    def swap_color_channels(self):
        """
        Exchange the red and blue channels; green stays in place.

        [:, :, [2, 1, 0]] means: all rows, all columns, and take the colour
        layers in the order 2, 1, 0 -> [B, G, R] becomes [R, G, B].

        Returns:
            BGR uint8 image with red and blue swapped.
        """
        swapped_image = self.image[:, :, [2, 1, 0]].copy()
        return swapped_image

    # ------------------------------------------------------------------
    # Task 5 - foliage detection (thresholding)
    # ------------------------------------------------------------------
    def foliage_detection(self):
        """
        Mark "green" pixels: ON if G >= 50 and R < 50 and B < 50.

        Returns:
            Single-channel uint8 binary image: 255 = foliage, 0 = other.
        """
        B = self.image[:, :, 0]
        G = self.image[:, :, 1]
        R = self.image[:, :, 2]

        mask = (G >= 50) & (R < 50) & (B < 50)   # True only where all 3 hold

        foliage_image = np.zeros(G.shape, dtype=np.uint8)   # all black
        foliage_image[mask] = 255                            # paint leaves white
        return foliage_image

    # ------------------------------------------------------------------
    # Task 6 - translation
    # ------------------------------------------------------------------
    def shift_image(self):
        """
        Move the image 200 px right and 100 px down; empty area = 0 (black).

        M = [[1, 0, tx], [0, 1, ty]] is a pure translation:
            new_x = x + tx,  new_y = y + ty
        cv2.warpAffine fills uncovered pixels with 0 by default.

        Returns:
            BGR uint8 image of the same size.
        """
        h, w = self.image.shape[:2]
        M = np.float32([[1, 0, 200],     # move right by 200
                        [0, 1, 100]])    # move down by 100
        shifted_image = cv2.warpAffine(self.image, M, (w, h))   # size is (w, h)
        return shifted_image

    # ------------------------------------------------------------------
    # Task 7 - rotate 90 degrees clockwise
    # ------------------------------------------------------------------
    def rotate_image(self):
        """
        Rotate 90 degrees clockwise. A w x h image becomes h x w.

        Returns:
            BGR uint8 image with height and width swapped.
        """
        rotated_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image

    # ------------------------------------------------------------------
    # Task 8 - similarity transform with inverse mapping
    # ------------------------------------------------------------------
    def similarity_transform(self, scale, theta, shift):
        """
        Scale -> rotate (clockwise) -> translate, about the origin, written
        by hand with nearest-neighbour inverse mapping.

        Forward model (input point p -> output point q):
            q = R * (s * p) + t,   R = [[cos, -sin], [sin, cos]]
        Inverse mapping visits EVERY output pixel and undoes the moves in
        reverse order, so no output pixel is left empty:
            p = R^T * (q - t) / s

        Args:
            scale: scale factor (float), e.g. 2.0
            theta: rotation angle in degrees, clockwise (float), e.g. 45.0
            shift: [tx, ty] translation in pixels, e.g. [100, 100]

        Returns:
            BGR uint8 image, same size as the input.
        """
        h, w = self.image.shape[:2]
        t = np.deg2rad(theta)                      # cos/sin need radians
        cos_t, sin_t = np.cos(t), np.sin(t)
        tx, ty = float(shift[0]), float(shift[1])

        # (x, y) coordinates of every OUTPUT pixel
        yq, xq = np.mgrid[0:h, 0:w].astype(np.float64)

        # 1) undo translation
        xs = xq - tx
        ys = yq - ty

        # 2) undo rotation (R transpose) and 3) undo scale
        xp = (cos_t * xs + sin_t * ys) / scale
        yp = (-sin_t * xs + cos_t * ys) / scale

        # nearest neighbour: round to the closest whole source pixel
        xp = np.round(xp).astype(int)
        yp = np.round(yp).astype(int)

        # keep only source positions that lie inside the input image
        valid = (xp >= 0) & (xp < w) & (yp >= 0) & (yp < h)

        transformed_image = np.zeros_like(self.image)          # black canvas
        transformed_image[valid] = self.image[yp[valid], xp[valid]]
        return transformed_image

    # ------------------------------------------------------------------
    # Task 9 - grayscale conversion
    # ------------------------------------------------------------------
    def convert_to_grayscale(self):
        """
        gray = (3R + 6G + 1B) / 10, rounded to the nearest integer.

        Channels are converted to float first so the weighted sum cannot
        overflow uint8 (e.g. 6 * 200 = 1200 > 255).

        Returns:
            Single-channel uint8 grayscale image.
        """
        B = self.image[:, :, 0].astype(np.float64)
        G = self.image[:, :, 1].astype(np.float64)
        R = self.image[:, :, 2].astype(np.float64)
        gray_image = np.round((3 * R + 6 * G + 1 * B) / 10).astype(np.uint8)
        return gray_image

    # ------------------------------------------------------------------
    # Task 10 - image moments of a binary image
    # ------------------------------------------------------------------
    def compute_moments(self):
        """
        First- and second-order moments of the binary image
        (white pixels = 1, black = 0).

            m00            area (number of white pixels)
            m10, m01       sum of x, sum of y
            x_bar, y_bar   centroid = (m10/m00, m01/m00)
            mu20           sum (x - x_bar)^2            spread left-right
            mu02           sum (y - y_bar)^2            spread up-down
            mu11           sum (x - x_bar)(y - y_bar)   diagonal tilt

        Returns:
            (m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11)
        """
        img = (self.binary_image > 0).astype(np.float64)   # 1 = white, 0 = black
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]                          # row / column grids

        # Standard (raw) moments
        m00 = np.sum(img)
        m10 = np.sum(x * img)
        m01 = np.sum(y * img)

        # Centroid
        x_bar = m10 / m00
        y_bar = m01 / m00

        # Second-order centralized moments
        mu20 = np.sum(((x - x_bar) ** 2) * img)
        mu02 = np.sum(((y - y_bar) ** 2) * img)
        mu11 = np.sum((x - x_bar) * (y - y_bar) * img)

        # print("First-Order Moments:")
        # print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        # print("Centralized Moments:")
        # print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        # print("Second-Order Centralized Moments:")
        # print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    # ------------------------------------------------------------------
    # Task 11 - orientation, eccentricity and best-fitting ellipse
    # ------------------------------------------------------------------
    def compute_orientation_and_eccentricity(self):
        """
        Shape analysis from the Task 10 moments.

            a, b, c       = mu20/m00, mu11/m00, mu02/m00
            lambda1 >= lambda2  eigenvalues of [[a, b], [b, c]]
            orientation   = degrees(0.5 * arctan(2*mu11 / (mu20 - mu02)))
                            (clockwise from +x, since y points down)
            eccentricity  = sqrt((lambda1 - lambda2) / lambda1)
                            (0 = circle, close to 1 = long and thin)
            full axis     = 4 * sqrt(lambda); cv2.ellipse gets half of it

        The ellipse is drawn in red (0, 0, 255), 1 px thick, on a copy of
        the BGR image so it appears on top of the white outline.

        Returns:
            (orientation_degrees, eccentricity, image_with_ellipse)
        """
        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()

        orientation = np.degrees(0.5 * np.arctan(2 * mu11 / (mu20 - mu02)))

        a, b, c = mu20 / m00, mu11 / m00, mu02 / m00
        lam = np.linalg.eigvalsh(np.array([[a, b], [b, c]]))   # ascending
        lambda2, lambda1 = lam[0], lam[1]                       # lambda1 = larger

        eccentricity = np.sqrt((lambda1 - lambda2) / lambda1)

        axes = (int((4 * np.sqrt(lambda1)) // 2),               # half major axis
                int((4 * np.sqrt(lambda2)) // 2))               # half minor axis
        center = (int(x_bar), int(y_bar))

        glasses_with_ellipse = self.image.copy()
        cv2.ellipse(glasses_with_ellipse, center, axes, orientation,
                    0, 360, (0, 0, 255), 1)

        # print(f"Orientation: {orientation} degrees")
        # print(f"Eccentricity: {eccentricity}")

        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":
    # Runs every task silently (no prints, no files) - safe for autograders.
    # Use run_demo.py to print values and save result images.
    here = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(here, "images")

    assignment = ComputerVisionAssignment(
        os.path.join(img_dir, "original_image.png"),
        os.path.join(img_dir, "binary_image.png"),
    )

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()

    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()

    glasses_path = os.path.join(img_dir, "glasses_outline.png")
    glasses_assignment = ComputerVisionAssignment(glasses_path, glasses_path)

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
