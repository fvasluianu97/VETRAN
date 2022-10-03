import math
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
from edge_loss import edge_loss
from loss import PerceptualLossModule
from torch.utils.tensorboard import SummaryWriter
from parameters import params
from skimage.io import imsave
from vetran import Generator
from functions import hinge_loss
from torch.nn.functional import mse_loss
from torchvision import transforms
from torchvision.utils import save_image


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

        m.weight.data.normal_(0, math.sqrt(2.0/n))
        
        if m.bias is not None:
            m.bias.data.uniform_(0, 255)


class System:
    def __init__(self, device, fast_eval=False):
        print("Used parameters:")
        for key, value in params.items():
            print(key, ' : ', value)
        print("\n\n")

        # assign device to member variable for usage in member functions
        self.device = device
        if params["alpha_perc"] > 0.0:
            self.pl = PerceptualLossModule(device)

        if fast_eval:
            # load generator model and set to eval mode
            self.g = Generator(self.device).to(device).eval()  # generator G

            # set evaluation mode
            self.fast_eval = True

        else:
            os.makedirs("{}/logs/checkpoints".format(params["model_name"]), exist_ok=True)
            self.monitor = SummaryWriter("{}/logs/tensorboard".format(params["model_name"]))

            # initialize models
            self.g = Generator(self.device).to(device)

            # setup optimizers
            self.optimizer_g = torch.optim.Adam(self.g.parameters(), lr=params["lr"])
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda = lambda iteration: params['lambda init'] ** iteration) 
            # set iteration
            self.iter = 0

            # set evaluation mode
            self.fast_eval = False

        # set current model string
        self.model = None

        # load last checkpoint if available
        if os.path.exists("{}/logs/checkpoints/last.txt".format(params["model_name"])):
            self.load_checkpoint()
    
    def train_step(self, x, y):
        self.optimizer_g.zero_grad()        
        g_x, g_sx, g_dx, y_s, y_d = self.g(x, y)

        mse = torch.mean(F.mse_loss(y, g_x))
        psnr = 10 * torch.log10(255**2 / mse)
        
        x_mse = torch.mean(F.mse_loss(y, x))
        x_psnr = 10 * torch.log10(255**2/x_mse)

        if params['loss'] == 'chrb':
            reconstruction_loss = 1/x.shape[1] * (params['gamma'] * torch.sqrt(F.mse_loss(g_x, y))
                                            + params['alpha'] * torch.sqrt(F.mse_loss(g_sx, y_s))
                                            + params['beta'] * torch.sqrt(F.mse_loss(g_dx, y_d)))
        else:
            reconstruction_loss = F.mse_loss(g_x, y) + F.mse_loss(g_sx, y_s) + F.mse_loss(g_dx, y_d)
            reconstruction_loss += params['delta'] * edge_loss(g_x.squeeze(0), y.squeeze(0))

        if params["alpha_perc"] > 0:
            perceptual_loss = self.pl.compute_perceptual_loss(g_x.squeeze(0), y.squeeze(0))
        else:
            perceptual_loss = 0.0

        loss = params["alpha_chrb"] * reconstruction_loss +  params["alpha_perc"] * perceptual_loss


        # monitor losses
        self.monitor.add_scalar("Generator/Train Loss", loss.item(), self.iter)

        loss.backward()
        if params["grad_clipping"] == True:
            torch.nn.utils.clip_grad_norm_(self.g.parameters(), params["clip value"])
        
        self.optimizer_g.step()

        self.iter += 1 
        # control LR to avoid saturating
        if params['sched'] == True and self.iter % params["lambda step"] == 0:
            self.lr_scheduler.step()

        return loss.item(), mse, psnr, x_mse, x_psnr

    def save_image(self, img_tensor, img_path, normalize=True):
        img_array = np.moveaxis(img_tensor.cpu().detach().numpy(), 0, -1)
        img_array = np.rint(np.clip(img_array, 0, 255)).astype(np.dtype('uint8'))
        imsave(img_path, img_array)
    
    def val_step(self, x, y, mode="validation", idx=0, dir=None):
        with torch.no_grad():
            start_time = time.time()
            g_x, g_sx, g_dx, y_s, y_d = self.g(x, y)
            end_time = time.time()

            if params['verbose'] == True:
                print("Inference time:{}".format(end_time - start_time))

            s_cmse = 0
            for i in range(g_x.shape[1]):
                gx = g_x.squeeze(0)[i, :, :, :]
                ix = x.squeeze(0)[i, :, :, :]
                iy = y.squeeze(0)[i, :, :, :]
                MSE = mse_loss(gx, iy).item()

                _, H, W = gx.shape
                gxm = gx.mean(dim=(1, 2))
                xm = ix.mean(dim=(1, 2))
                diff = gxm - xm
                diff_expand = diff.unsqueeze(-1).unsqueeze(-1).expand_as(gx)
                c_g_x = gx - diff_expand
                
                c_MSE = mse_loss(c_g_x, iy).item()
                s_cmse += c_MSE

                if mode == "testing_full":
                    if params["challenge"] == True:
                        frame_idx = idx * params["test sequence length"] + i
                        if (frame_idx + 1) % 10 == 0:
                            save_image(g_x, "{}/{}.png".format(dir, str(frame_idx + 1).zfill(3)), normalize=True) 
                    else:
                        self.save_image(gx, "{}/{}_generated.png".format(dir, str(idx * params["test sequence length"]  + i).zfill(3)), normalize=True)
                        self.save_image(ix, "{}/{}_compressed.png".format(dir, str(idx * params["test sequence length"] + i).zfill(3)), normalize=True)
                        self.save_image(iy, "{}/{}_target.png".format(dir, str(idx * params["test sequence length"] + i).zfill(3)), normalize=True)
                        self.save_image(c_g_x, "{}/{}_corrected.png".format(dir, str(idx * params["test sequence length"] + i).zfill(3)), normalize=True) 
                    
            elapsed = time.time() - start_time
            if params['verbose'] == True:
                print("Correction time: {}".format(elapsed))

        mse = torch.mean(F.mse_loss(y, g_x))
        psnr = 10 * torch.log10(255**2 / mse)
        
        seq_len = x.shape[1]
        cmse = s_cmse / seq_len
        cpsnr = 10 * torch.log10(255**2 / torch.tensor(cmse)) 
       
        x_mse = torch.mean(F.mse_loss(y, x))
        x_psnr = 10 * torch.log10(255**2/x_mse)

        return mse, cmse, psnr, cpsnr, x_mse, x_psnr

    def eval(self, x, y):

        if self.fast_eval:
            with torch.no_grad():
                out = self.g(x, y)

            return out

        else:
            self.g.eval()
            with torch.no_grad():
                out = self.g(x, y)
            self.g.train()

            return out        

    def save_checkpoint(self, name=None):

        print("Saving checkpoint...")
        if name is None:
            save_string = "{}/logs/checkpoints/".format(params["model_name"]) + str(self.iter).zfill(7) + ".pt"
        else:
            save_string = "{}/logs/checkpoints/".format(params["model_name"]) + name + "_" + str(self.iter).zfill(7) + ".pt"

        torch.save({
            "g": self.g.state_dict(),
            "optimizer g": self.optimizer_g.state_dict(),
            "lr sched": self.lr_scheduler.state_dict(),
            "iteration": self.iter}, save_string)

        # set "last checkpoint" file
        with open("{}/logs/checkpoints/last.txt".format(params["model_name"]), "w") as f:
            f.write(save_string)

        print("Saved checkpoint at: ", save_string)

    def load_checkpoint(self, file_name="last"):

        if file_name == "last":
            print("Loading last checkpoint...")
            with open("{}/logs/checkpoints/last.txt".format(params["model_name"]), "r") as f:
                path = f.readline().strip()
        else:
            print("Loading checkpoint: ", "logs/checkpoints/" + file_name)
            path = "{}/logs/checkpoints/".format(params["model_name"]) + file_name

        checkpoint = torch.load(path, map_location=self.device)

        self.g.load_state_dict(checkpoint["g"], strict=False)

        if not self.fast_eval:
            try:
                self.optimizer_g.load_state_dict(checkpoint["optimizer g"])
                self.lr_scheduler.load_state_dict(checkpoint["lr sched"])
                print("Loaded checkpoint: ", path)
            except ValueError:
                print("Impossible to load the optimizer state. Training using the initialization")
            finally:
                self.iter = checkpoint["iteration"]

        # set model string
        self.model = path[-10:-3]

    def set_learningrates(self, lr):
        for g in self.optimizer_g.param_groups:
            g["lr"] = lr
            print("New learning rate for g optimizer: ", g["lr"] )

