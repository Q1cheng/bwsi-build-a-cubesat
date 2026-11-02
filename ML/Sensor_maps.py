import cv2
import torch
import torch.nn.functional as F
import numpy as np

def rgb_to_gray(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return (gray.astype(np.float32) / 255.0)

def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def random_gaussian_blur(gray_hw, p=0.5, sigma_min=0.2, sigma_max=1.2):
    if np.random.rand() > p:
        return gray_hw

    sigma = np.random.uniform(sigma_min, sigma_max)
    blurred = cv2.GaussianBlur(gray_hw, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return blurred.astype(np.float32)

def make_sensor_maps(gray_hw):
    #channels = [gray, grad, var, hf, lap]
    gray = torch.from_numpy(gray_hw).float().unsqueeze(0)  # (1, H, W)

    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gx = F.pad(gx, (0, 1, 0, 0))

    gy = gray[:, 1:, :] - gray[:, :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1))

    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)

    gray_batched = gray.unsqueeze(0)  # (1, 1, H, W)

    mean = F.avg_pool2d(gray_batched, kernel_size=7, stride=1, padding=3).squeeze(0)
    mean2 = F.avg_pool2d(gray_batched * gray_batched, kernel_size=7, stride=1, padding=3).squeeze(0)

    var = torch.clamp(mean2 - mean * mean, min=0.0)
    hf = torch.abs(gray - mean)

    lap_kernel = torch.tensor(
        [[0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]],
        dtype=gray_batched.dtype,
        device=gray_batched.device
    ).view(1, 1, 3, 3)

    lap = F.conv2d(gray_batched, lap_kernel, padding=1).squeeze(0)

    Maps = torch.cat([gray, grad, var, hf, lap], dim=0)

    return Maps