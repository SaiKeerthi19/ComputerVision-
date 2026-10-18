#!/usr/bin/env python
import numpy as np
import cv2
import scipy
import scipy.signal
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helper functions (plain add/multiply + double for-loop over the pixels)
# ---------------------------------------------------------------------------
def convolve_vertical(image, kernel):
  """Slide a 3x1 (vertical) kernel over the image. Zero padding. No flip:
  kernel[0] multiplies the pixel ABOVE, kernel[1] the pixel itself,
  kernel[2] the pixel BELOW."""
  img = image.astype(np.float64)
  h, w = img.shape
  out = np.zeros((h, w), dtype=np.float64)
  for y in range(h):
    for x in range(w):
      total = 0.0
      for i in range(3):            # i = 0,1,2  ->  row offset -1,0,+1
        yy = y + i - 1
        if 0 <= yy < h:             # outside the image = 0 (zero padding)
          total = total + kernel[i] * img[yy, x]
      out[y, x] = total
  return out


def convolve_horizontal(image, kernel):
  """Slide a 1x3 (horizontal) kernel over the image. Zero padding. No flip:
  kernel[0] multiplies the pixel on the LEFT, kernel[2] the pixel on the RIGHT."""
  img = image.astype(np.float64)
  h, w = img.shape
  out = np.zeros((h, w), dtype=np.float64)
  for y in range(h):
    for x in range(w):
      total = 0.0
      for i in range(3):            # column offset -1,0,+1
        xx = x + i - 1
        if 0 <= xx < w:
          total = total + kernel[i] * img[y, xx]
      out[y, x] = total
  return out


def to_uint8_signed(image):
  """pout = clamp(2 * pin + 127) -> uint8 (keeps negative values visible)."""
  return np.clip(2.0 * image + 127.0, 0, 255).astype(np.uint8)


class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (red)
    fill_color = (0, 0, 255)  # (B, G, R)
    # Create a copy of the input image to keep the original image unchanged
    output_image = self.ant_img.copy()
    h, w = output_image.shape[:2]

    x0, y0 = seed                       # seed is (x, y) -> image[y, x]
    old_color = output_image[y0, x0].copy()
    if np.array_equal(old_color, fill_color):
      return output_image               # nothing to do

    # Define a stack for floodfill
    stack = [(x0, y0)]
    # 8-connected neighbours (including diagonals)
    neighbors = [(-1, -1), (0, -1), (1, -1),
                 (-1,  0),          (1,  0),
                 (-1,  1), (0,  1), (1,  1)]

    while stack:
      x, y = stack.pop()
      if x < 0 or x >= w or y < 0 or y >= h:
        continue                        # outside the image
      if not np.array_equal(output_image[y, x], old_color):
        continue                        # wall, or already painted
      output_image[y, x] = fill_color   # paint it
      for dx, dy in neighbors:          # push all 8 neighbours
        stack.append((x + dx, y + dy))

    #cv2.imwrite('floodfill.png', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = [0.25, 0.5, 0.25]  # 1D Gaussian kernel = 0.25 * [1 2 1]
    image = self.cat_eye.astype(np.float64)
    self.blurred_images = []
    for i in range(5):
        # Apply convolution (vertical 3x1, then horizontal 1x3)
        image = convolve_vertical(image, kernel)
        image = convolve_horizontal(image, kernel)

        # Store the blurred image (kept as float for precision)
        self.blurred_images.append(image)

        #cv2.imwrite(f'gaussian_blur_{i}.png', np.clip(image, 0, 255).astype(np.uint8))
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    smooth = [0.25, 0.5, 0.25]   # horizontal direction: 0.25 * [1 2 1]
    diff = [0.5, 0.0, -0.5]      # vertical direction:   0.5 * [1 0 -1]

    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution
      image = convolve_horizontal(self.blurred_images[i], smooth)
      image = convolve_vertical(image, diff)
      image = to_uint8_signed(image)

      self.vDerive_images.append(image)
      #cv2.imwrite(f'vertical_{i}.png', image)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels
    smooth = [0.25, 0.5, 0.25]   # vertical direction:   0.25 * [1 2 1]
    diff = [0.5, 0.0, -0.5]      # horizontal direction: 0.5 * [1 0 -1]

    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):

      # Apply horizontal and vertical convolution
      image = convolve_vertical(self.blurred_images[i], smooth)
      image = convolve_horizontal(image, diff)
      image = to_uint8_signed(image)

      self.hDerive_images.append(image)
      #cv2.imwrite(f'horizontal_{i}.png', image)
    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
      # Undo the scale/offset to get back signed gradients
      gy = (vimg.astype(np.float64) - 127.0) / 2.0
      gx = (himg.astype(np.float64) - 127.0) / 2.0
      # Manhattan magnitude, scaled by 4 and clamped to uint8
      image = np.clip(4.0 * (np.abs(gx) + np.abs(gy)), 0, 255).astype(np.uint8)
      self.gdMagnitute_images.append(image)
      #cv2.imwrite(f'gradient_{i}.png', image)
    return self.gdMagnitute_images

  def scipy_convolve(self):
    # Define the 2D kernel (vertical Gaussian derivative = Sobel)
    smooth = np.array([0.25, 0.5, 0.25])   # along x
    diff = np.array([0.5, 0.0, -0.5])      # along y
    kernel_2d = np.outer(diff, smooth)     # 3x3

    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution (zero padding, same size as input)
      image = scipy.signal.convolve2d(self.blurred_images[i], kernel_2d,
                                      mode='same', boundary='fill', fillvalue=0)
      image = to_uint8_signed(image)
      self.scipy_smooth.append(image)
      #cv2.imwrite(f'scipy_smooth_{i}.png', image)
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D convolve (full mode -> length N + M - 1)
      out = np.convolve(np.array(out, dtype=np.int64),
                        np.array(box_filter, dtype=np.int64), mode='full')

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
    
