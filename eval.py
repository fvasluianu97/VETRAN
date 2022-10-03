import sys
import subprocess
from parameters import params
import numpy as np
import torch
from system import System
from loader_val import Loader as Loader_val
import time


loader_val = Loader_val()


def eval(system, device):
    print("Evaluation...", system.iter)
    avg_mse, avg_psnr = [], []
    avg_cmse, avg_cpsnr = [], []
    x_avg_mse, x_avg_psnr = [], []

    for i, (x_val, y_val) in enumerate(loader_val.dataloader):            
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        mse, cmse, psnr, cpsnr, x_mse, x_psnr = system.val_step(x_val, y_val)
        avg_mse.append(mse.item())
        avg_psnr.append(psnr.item())

        avg_cmse.append(cmse)
        avg_cpsnr.append(cpsnr.item())

        x_avg_mse.append(x_mse.item())
        x_avg_psnr.append(x_psnr.item())


    # average over all validation sequences
    total_mse = np.mean(avg_mse)
    total_psnr = np.mean(avg_psnr)

    total_cmse = np.mean(avg_cmse)
    total_cpsnr = np.mean(avg_cpsnr)
    
    total_x_mse = np.mean(x_avg_mse)
    total_x_psnr = np.mean(x_avg_psnr)

    print("validation MSE: ", total_mse, " CMSE: ", total_cmse, " PSNR: ", total_psnr, " CPSNR: ", total_cpsnr, "X_MSE: ", total_x_mse, "X_PSNR: ", total_x_psnr)
    
    return total_mse, total_cmse, total_psnr, total_cpsnr
