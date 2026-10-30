import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        # Q1a: OpenCV image (BGR, uint8, H x W x 3) -> RGB float32 torch tensor (H x W x 3)
        rgb = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)        # swap BGR -> RGB
        torch_img = torch.from_numpy(rgb).to(torch.float32)   # numpy -> torch, as float32
        return torch_img

    def brighten(self, torch_img):
        # Q1b: add 100.0 to every value (float32, so nothing wraps around)
        bright_img = torch_img + 100.0
        return bright_img

    def saturation_arithmetic(self, img):
        # Q1c: RGB, add 100 in int16, clamp to 0..255, then back to uint8
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        wide = torch.from_numpy(rgb).to(torch.int16)          # wider type: can hold 355
        wide = wide + 100
        saturated_img = torch.clamp(wide, 0, 255).to(torch.uint8)
        return saturated_img

    def add_noise(self, torch_img):
        # Q2: Gaussian noise, mean 0, sigma 100 gray levels; return float32 in 0..1
        noise = torch.randn(torch_img.shape) * 100.0           # randn = mean 0, std 1
        noisy = torch_img.to(torch.float32) + noise
        noisy_img = torch.clamp(noisy / 255.0, 0.0, 1.0).to(torch.float32)
        return noisy_img

    def normalization_image(self, img):
        # Q3a: subtract the image's own per-channel mean, divide by its std (float64, RGB)
        # No clamping -> result has zero mean and unit std per channel
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(torch.float64)
        mean = x.mean(dim=(0, 1))                              # 3 values: R, G, B
        std = x.std(dim=(0, 1))                                # 3 values: R, G, B
        image_norm = (x - mean) / std                          # broadcasting over H x W
        return image_norm

    def Imagenet_norm(self, img):
        # Q3b: scale to 0..1, normalize with ImageNet mean/std (RGB, float64), clamp to [0, 1]
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(torch.float64) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)
        ImageNet_norm = torch.clamp((x - mean) / std, 0.0, 1.0)
        return ImageNet_norm

    def dimension_rearrange(self, img):
        # Q4: H x W x C  ->  N x C x H x W  (N = 1), float32, RGB
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(torch.float32)            # (H, W, C)
        rearrange = x.permute(2, 0, 1).unsqueeze(0)            # (C, H, W) -> (1, C, H, W)
        return rearrange

    def stride(self, img):
        # Q5: 3x3 Scharr_x filter, zero padding 1, stride 2, on a grayscale image
        scharr_x = torch.tensor([[-3.0, 0.0, 3.0],
                                 [-10.0, 0.0, 10.0],
                                 [-3.0, 0.0, 3.0]], dtype=torch.float32)
        scharr_x = torch.flip(scharr_x, dims=[0, 1])           # true convolution flips the kernel
        x = torch.from_numpy(img).to(torch.float32)            # (H, W)
        x = x.unsqueeze(0).unsqueeze(0)                        # (1, 1, H, W) for conv2d
        k = scharr_x.unsqueeze(0).unsqueeze(0)                 # (1, 1, 3, 3)
        out = torch.nn.functional.conv2d(x, k, stride=2, padding=1)
        stride_imgage = out.squeeze(0).squeeze(0).type(torch.FloatTensor)   # back to 2-D
        return stride_imgage


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    img_cat = cv.imread("cat_eye.jpg", cv.IMREAD_GRAYSCALE)
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    stride_imgage = assign.stride(img_cat)
