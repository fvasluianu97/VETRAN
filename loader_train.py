import os
from parameters import params
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random
from skimage.io import imread
from skimage.transform import rotate, resize
import numpy as np
import matplotlib.pyplot as plt


# Dataset
class IntVID(Dataset):
    def __init__(self):

        root = params["dataset root"]
        self.videos = params["videos_train"]
        self.seq_len = params["sequence length"]
        self.cropsize_h = params["crop size h"]
        self.cropsize_w = params["crop size w"] 

        self.file_list_input = []
        self.file_list_target = []
        for i in range(params["train_startswith"], params["train_startswith"] + self.videos):
            self.file_list_input.append(sorted(glob.glob(root + "TrainingSourceDomain{}/".format(params['bitrate']) + str(i).zfill(3) + "/*.png")))
            self.file_list_target.append(sorted(glob.glob(root + "TrainingTargetDomain/" + str(i).zfill(3) + "/*.png")))
        
    

    def __len__(self):
        return 10**6

    def __getitem__(self, idx):
        index = idx % self.videos
        input_list = self.file_list_input[index]
        target_list = self.file_list_target[index]

        vid_len = len(input_list)
        try:
            seq_start = random.randint(0, vid_len - self.seq_len)
        except ValueError:
            print("ERROR: ", vid_len - self.seq_len)
            seq_start = 0

        # INPUT ####################################
        seq_input = []
        for i in range(self.seq_len):
            input = imread(input_list[seq_start + i])
            seq_input.append(input)

        # TARGET ####################################
        seq_target = []
        for i in range(self.seq_len):
            target = imread(target_list[seq_start + i])
            seq_target.append(target)

        x = np.moveaxis(np.array(seq_input, dtype=np.float32), -1, -3)
        y = np.moveaxis(np.array(seq_target, dtype=np.float32), -1, -3)
        
        cropsize_h = min(x.shape[-2], self.cropsize_h)
        cropsize_w = min(x.shape[-1], self.cropsize_w)

        
        # random crop
        if params['crop'] == True:
            try:
                rand_h = random.randint(0, x.shape[-2] - cropsize_h)
                rand_w = random.randint(0, x.shape[-1] - cropsize_w)
            except ValueError:
                print(x.shape, y.shape)
                rand_h = (x.shape[-2] - cropsize_h) // 2
                rand_w = (x.shape[-1] - cropsize_w) // 2
        
            x = x[..., rand_h: rand_h + cropsize_h, rand_w: rand_w + cropsize_w]
            y = y[..., rand_h: rand_h + cropsize_h, rand_w: rand_w + cropsize_w]

        return torch.from_numpy(x), torch.from_numpy(y)


# Dataloader
class Loader:
    def __init__(self):

        self.dataset = IntVID()
        self.epoch_size = self.dataset.videos
        self.batch_size = params["bs"]
        self.shuffle = True
        self.num_workers = params["number of workers"]

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers,
                                     pin_memory=True)


# # testing
if __name__ == '__main__':
    loader = Loader()
    idx = 0
    mean = []
    while True:
        for sample_x, sample_y in loader.dataloader:
            print(sample_x.shape, sample_y.shape)
            
