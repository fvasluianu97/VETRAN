import torch
import numpy as np
import torch.nn.functional as fnn


def edge_loss(out, target, device=torch.device("cuda:0")):
    x_filter = np.array([np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) for i in range(3)])
    y_filter = np.array([np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) for i in range(3)])
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0)

    d1x = fnn.conv2d(out, weight=weights_x.to(device))
    d2x = fnn.conv2d(target, weight=weights_x.to(device))
    d1y = fnn.conv2d(out, weight=weights_y.to(device))
    d2y = fnn.conv2d(target, weight=weights_y.to(device))

    g_1 = torch.sqrt(torch.pow(d1x, 2) + torch.pow(d1y, 2))
    g_2 = torch.sqrt(torch.pow(d2x, 2) + torch.pow(d2y, 2))

    return torch.mean((g_1 - g_2).pow(2)).item()
