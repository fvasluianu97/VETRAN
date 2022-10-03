import sys
import subprocess
from parameters import params
import numpy as np
import torch
from system import System
from loader_test import Loader as Loader_test

loader_test = Loader_test()


def test(system, device, dirname=None):
    print("Testing...", system.iter)
    avg_mse, avg_psnr = [], []
    avg_x_mse, avg_x_psnr = [], []
    avg_cmse, avg_cpsnr = [], []

    for i, (x_test, y_test) in enumerate(loader_test.dataloader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        mse, cmse, psnr, cpsnr, x_mse, x_psnr = system.val_step(x_test, y_test, mode="testing", idx=i, dir=dirname)
        if params["verbose"] == True: # print metrics by batch also
            print("mse: ", mse.item(), "cmse: ", cmse,  "psnr: ", psnr.item(), "/x_mse: ", x_mse.item(), "x_psnr: ", x_psnr.item())
        
        avg_cmse.append(cmse)
        avg_cpsnr.append(cpsnr)

        avg_mse.append(mse.item())
        avg_psnr.append(psnr.item())

        avg_x_mse.append(x_mse.item())
        avg_x_psnr.append(x_psnr.item())

    # average over all validation sequences
    total_mse = np.mean(avg_mse)
    total_cmse = np.mean(avg_cmse)
    total_psnr = np.mean(avg_psnr)
    total_cpsnr = np.mean(avg_cpsnr)
    total_x_mse = np.mean(avg_x_mse)
    total_x_psnr = np.mean(avg_x_psnr)

    print("Test - MSE: ", total_mse, " CMSE: ", total_cmse,  " PSNR: ", total_psnr, " CPSNR: ", total_cpsnr, " X_MSE: ", total_x_mse, " X_PSNR: ", total_x_psnr)

    return total_mse, total_cmse, total_psnr, total_cpsnr
