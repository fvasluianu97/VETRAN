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
class IntVIDFull(Dataset):
    def __init__(self, tst_idx=0):

        root = params["dataset root"]
        self.videos = params["videos_test"]
        self.seq_len = params["test sequence length"]
        self.vid_idx = tst_idx
        
        self.cropsize_h = params["crop size h"] // params["crop factor"]
        self.cropsize_w = params["crop size w"] // params["crop factor"]
        self.file_list_input = []
        self.file_list_target = []
        
        for i in range(params["test_startswith"], params["test_startswith"] + self.videos):
            self.file_list_input.append(sorted(glob.glob(root + "TestSource{}/".format(params['suffix']) + str(i).zfill(3) + "_*.png")))
            
            if params["challenge"] == False:
                self.file_list_target.append(sorted(glob.glob(root + "TestTarget{}/".format(params['suffix']) + str(i).zfill(3) + "_*.png")))
        
    
    def __len__(self):
        num_batches = len(self.file_list_input[self.vid_idx])//self.seq_len
        if num_batches * self.seq_len < len(self.file_list_input[self.vid_idx]):
            num_batches += 1

        return num_batches

    def __getitem__(self, idx):
        input_list = self.file_list_input[self.vid_idx]
        
        if params["challenge"] == False:
            target_list = self.file_list_target[self.vid_idx]

        vid_len = len(input_list)
        seq_start = self.seq_len * idx
        seq_len = min(self.seq_len, vid_len - seq_start)

        # INPUT #################################
        seq_input = []
        for i in range(seq_len):
            input = imread(input_list[seq_start + i])
            seq_input.append(input)

        x = np.moveaxis(np.array(seq_input, dtype=np.float32), -1, -3)
        
        # TARGET #################################
        if params["challenge"] == False:
            seq_target = []
            for i in range(seq_len):
                input = imread(target_list[seq_start + i])
                seq_target.append(input)

            y = np.moveaxis(np.array(seq_target, dtype=np.float32), -1, -3)
        
        if params["challenge"] == False:
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            return torch.from_numpy(x)


# Dataloader
class Loader:
    def __init__(self, tst_idx=0):
        self.dataset = IntVIDFull(tst_idx=tst_idx)
        self.epoch_size = self.dataset.videos
        self.batch_size = 1
        self.shuffle = False
        self.num_workers = params["test number of workers"]

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers,
                                     pin_memory=False)


if __name__ == '__main__':
    # # testing
    loader = Loader()
    idx = 0
    mean = []
    while True:
        for sample in loader.dataloader:
            print(sample[0].shape)
            disp = np.concatenate(np.moveaxis(sample[0].numpy(), 1, -1), axis=1)
            print(disp.shape, disp.max(), disp.min())

