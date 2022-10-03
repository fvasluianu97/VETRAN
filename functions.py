import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter as gf
from parameters import params


def hinge_loss(x, target):

    assert target == 0 or target == 1, "Target must be either 0 or 1!"

    if target == 1:
        return torch.clamp(params["margin"] - x, min=0), -x
    elif target == 0:
        return torch.clamp(params["margin"] + x, min=0), x


def shuffle_down(x, factor):
    # format: (B, C, H, W)
    # check for valid shapes
    b, c, h, w = x.shape

    assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

    n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
    n = n.permute(0, 3, 5, 1, 2, 4)
    n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))

    return n


def shuffle_up(x, factor):
    # format: (B, C, H, W)
    # check for valid shapes
    b, c, h, w = x.shape

    assert c % factor**2 == 0, "C must be a multiple of " + str(factor**2) + "!"

    n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
    n = n.permute(0, 3, 4, 1, 5, 2)
    n = n.reshape(b, int(c/(factor**2)), factor*h, factor*w)

    return n


def get_gaussian_kernel(k=5, sigma=2, channels=6, device="cuda:0"):
    x = torch.arange(k)
    x_g = x.repeat(k).view(k, k)
    y_g = x_g.t()
    xy_grid = torch.stack([x_g, y_g], dim=-1).float()

    mean = (k - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, k, k)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=k, padding=int(k/2), groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    gaussian_filter = gaussian_filter.to(device)

    return gaussian_filter

