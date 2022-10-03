import sys
import subprocess
from parameters import params
import numpy as np
import torch
from deploy_model import full_test_benchmark
from system import System
from loader_train import Loader as Loader_train
from eval import eval
from sample_batches import test
import time
import datetime
import os


start_time = str(datetime.datetime.now())
dirname = "{}/{}".format(params["results dir"], params["model_name"])
os.makedirs(dirname, exist_ok=True)


print("Available devices: {}".format(torch.cuda.device_count()))
if torch.cuda.is_available():
    device = torch.device(params["device"])
else:
    device = torch.device("cpu")

print("***************** device: ", device, " / system time: ", time.ctime(), "*****************")

# initialize trainer and loader objects
system = System(device)
loader_train = Loader_train()

# manually perform a testing step if resuming training
if system.iter > 0:
   psnr, cpsnr = full_test_benchmark(mode="testing", videos_test=params["videos_test"], system=system) 
   max_full_tst_psnr = max(psnr, cpsnr)
else:
    max_full_tst_psnr = 0


mse, cmse, psnr, cpsnr = eval(system, device)
print("Starting at mse {}/{}, psnr {}/{} ".format(mse, cmse, psnr, cpsnr))


# start training
start = time.time()
# train epoch
smse = 0
sxmse = 0
spsnr = 0
sxpsnr = 0
s_tloss = 0

# keep best stats
max_tst_psnr = 0

for epoch in range(params["num_epochs"]):
    for x, y in loader_train.dataloader:

        x = x.to(device)
        y = y.to(device)

        t_loss, mse, psnr, x_mse, x_psnr = system.train_step(x=x, y=y)
        s_tloss += t_loss
        smse += mse
        spsnr += psnr
        sxmse += x_mse
        sxpsnr += x_psnr

        if system.iter % 100 == 0:
            print(int(system.iter/loader_train.epoch_size), system.iter, (time.time() - start)/3600,
                  s_tloss/100, smse.item()/100, sxmse.item()/100, spsnr.item()/100, sxpsnr.item()/100)

            sys.stdout.flush()
            smse = 0
            sxmse = 0
            spsnr = 0
            sxpsnr = 0
            s_tloss = 0

        if system.iter % params["save interval"] == 0:
            system.save_checkpoint()

        try:
            if system.iter % params["eval interval"] == 0:
                mse, cmse, psnr, cpsnr  = eval(system, device)
        except Exception as e:
            print("exception occured during evaluation...", e)

        try:
            if system.iter % params["test interval"] == 0 and params["type"] != "deployment":
                mse, cmse, psnr, cpsnr = test(system, device, dirname)

                bpsnr = max(psnr, cpsnr)
                if bpsnr > max_tst_psnr:
                    max_tst_psnr = bpsnr
                    
                    psnr, cpsnr = full_test_benchmark(mode='testing', videos_test=params['videos_test'], system=system)
                    bpsnr = max(psnr, cpsnr)

                    if bpsnr > max_full_tst_psnr:
                        max_full_tst_psnr = bpsnr
                        system.save_checkpoint()

        except Exception as e:
            print("exception occured during testing...", e)

        if system.iter % params["full test interval"] == 0: 
            psnr, cpsnr = full_test_benchmark(mode="testing", videos_test=params['videos_test'], system=system)
            bpsnr = max(psnr, cpsnr)

            if bpsnr > max_full_tst_psnr:
                max_full_tst_psnr = bpsnr
                system.save_checkpoint()
        
        if params["server"] == "local" and (time.time() - start)/3600 > 47.5:
            break

system.save_checkpoint()


