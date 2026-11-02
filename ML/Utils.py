import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad(x):
    _, _, H, W = x.shape
    ph = (16 - H % 16) % 16
    pw = (16 - W % 16) % 16
    pad = (0, pw, 0, ph)
    return F.pad(x, pad, mode="reflect"), pad

def unpad(x, pad):
    _, pw, _, ph = pad
    H = x.shape[-2] - ph
    W = x.shape[-1] - pw
    return x[:, :, :H, :W]
