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
        self.videos = params["videos_val"]
        self.seq_len = params["eval sequence length"]

        self.cropsize_h = params["crop size h"]
        self.cropsize_w = params["crop size w"]

        self.file_list_input = []
        self.file_list_target = []
        for i in range(params["val_startswith"], params["val_startswith"] + self.videos):
            self.file_list_input.append(sorted(glob.glob(root + "ValidationSource{}/".format(params['bitrate']) + str(i).zfill(3) + "_*.png")))
            self.file_list_target.append(sorted(glob.glob(root + "ValidationTarget/" + str(i).zfill(3) + "_*.png")))

    def __len__(self):
        return self.videos

    def __getitem__(self, idx):

        index = idx
        input_list = self.file_list_input[index]
        target_list = self.file_list_target[index]

        vid_len = len(input_list)
        seq_start = np.random.randint(vid_len - self.seq_len - 1)

        # INPUT #################################
        seq_input = []
        for i in range(self.seq_len):

            input = imread(input_list[seq_start + i])
            seq_input.append(input)

        x = np.moveaxis(np.array(seq_input, dtype=np.float32), -1, -3)

        # TARGET #################################
        seq_target = []
        for i in range(self.seq_len):

            input = imread(target_list[seq_start + i])
            seq_target.append(input)

        y = np.moveaxis(np.array(seq_target, dtype=np.float32), -1, -3)

        #crop_h = (x.shape[-2] - self.cropsize_h) // 2
        #crop_W = (x.shape[-1] - self.cropsize_w) // 2
        
        #x = x[..., crop_h:crop_h + self.cropsize_h, crop_W:crop_W + self.cropsize_w]
        #y = y[..., crop_h:crop_h + self.cropsize_h, crop_W:crop_W + self.cropsize_w]
        
        
        return torch.from_numpy(x), torch.from_numpy(y)


# Dataloader
class Loader:
    def __init__(self):

        self.dataset = IntVID()
        self.epoch_size = self.dataset.videos
        self.batch_size = 1
        self.shuffle = False
        self.num_workers = params["eval number of workers"]

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=self.shuffle,
                                     num_workers=self.num_workers,
                                     pin_memory=False) # related to the allocator issue


# # testing
# loader = Loader()
# idx = 0
# mean = []
# while True:
#     for sample in loader.dataloader:
#         print(sample.shape)
#         disp = np.concatenate(np.moveaxis(sample[0].numpy(), 1, -1), axis=1)
#         print(disp.shape, disp.max(), disp.min())
#         plt.imshow(np.array(disp, dtype=np.uint8))
#         plt.show()
