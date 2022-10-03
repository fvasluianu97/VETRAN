from parameters import params
import numpy as np
import torch
from system import System
from loader_test_deployment import Loader as Loader_test
from eval import eval
from test_video import test
import time
import datetime
import os
import argparse


def full_test_benchmark(mode="testing", system=None, videos_test=16, data_dir='./local_results'):
    start_time = str(datetime.datetime.now())

    print("Available devices: {}".format(torch.cuda.device_count()))
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # initialize trainer and loader objects
    mses = []
    cmses = []
    psnrs = []
    cpsnrs = []
    xmses = []
    xpsnrs = []
    
    if system is None:
        system = System(device)

    assert videos_test <= params["videos_test"]

    for vid_idx in range(videos_test):
        dirname = "{}/{}/{}/{}".format(data_dir, params["model_name"], system.iter, str(vid_idx + params["test_startswith"]).zfill(3))
        if mode == "testing_full":
            os.makedirs(dirname, exist_ok=True)
        loader_test = Loader_test(vid_idx)
        mse, cmse, psnr, cpsnr, x_mse, x_psnr = test(system, device, mode, loader_test, dirname)
    
        system.g.state = None
        mses.append(mse)
        cmses.append(cmse)
        psnrs.append(psnr)
        cpsnrs.append(cpsnr)
        xmses.append(x_mse)
        xpsnrs.append(x_psnr)

    avg_mse = np.average(np.array(mses))
    avg_cmse = np.average(np.array(cmses))
    avg_psnr = np.average(np.array(psnrs))
    avg_cpsnr = np.average(np.array(cpsnrs))
    avg_xmse = np.average(np.array(xmses))
    avg_xpsnr = np.average(np.array(xpsnrs))

    print("Total average MSE/PSNR: {} - {}/{} - {} on input {}/{}".format(avg_mse, avg_cmse, avg_psnr, avg_cpsnr, avg_xmse, avg_xpsnr))
    return avg_psnr, avg_cpsnr


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='testing', help='Benchmark mode')
    parser.add_argument('--n_vids', type=int, default=16, help='Number of videos in the benchmark')
    parser.add_argument('--results_dir', default='./local_results', help='Path to save frames if mode=testing_full')
    opt = parser.parse_args()

    full_test_benchmark(mode=opt.mode, system=None, videos_test=opt.n_vids, data_dir=opt.results_dir )
