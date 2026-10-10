import torch
import torch.nn.functional as F


def compute_targets(gray):
    g = torch.from_numpy(gray).float()

    gx = g[:, :, 1:] - g[:, :, :-1]
    gx = F.pad(gx, (0, 1, 0, 0))

    gy = g[:, 1:, :] - g[:, :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1))

    grad = torch.sqrt(gx * gx + gy * gy)

    gb = g.unsqueeze(0)
    mean = F.avg_pool2d(gb, 7, 1, 3).squeeze(0)
    mean2 = F.avg_pool2d(gb * gb, 7, 1, 3).squeeze(0)
    var = torch.clamp(mean2 - mean * mean, min=0.0)

    hf = torch.abs(g - mean)

    smooth = 1.0 - torch.clamp(var / (var.max()), 0.0, 1.0)
    rough = torch.clamp((grad + hf) / ((grad + hf).max()), 0.0, 1.0)
    bright = torch.clamp(g, 0.0, 1.0)

    c = 100.0 + 900.0 * smooth
    phi = 30.0 + 40.0 * rough
    rho = 925.0 + 1025.0 * bright

    return torch.cat([c, phi, rho], dim=0).float()
