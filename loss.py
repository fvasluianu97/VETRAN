import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft

from PIL import Image, ImageFilter
from torchvision.models import vgg16


os.environ['TORCH_HOME'] = "./loaded_models/"

def gram_matrix(i_input):
    a, b, c, d = i_input.size()
    features = i_input.view(a * b, c * d)
    Gm = torch.mm(features, features.t())
    return Gm.div(a * b * c * d)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained = True).features)[:23]
        self.features = nn.ModuleList(features).eval()
    
    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 15, 22}:
                results.append(x)  
        return results

class PerceptualLossModule:
    def __init__(self, device):
        self.device = device
        self.model = Vgg16()
        self.criterion = torch.nn.MSELoss()
        self.model.to(self.device)
        self.criterion.to(self.device)
    
    def compute_color_loss(self, i_input, target):
        num_img = i_input.size()[0]
        color_loss = 0
        for i in range(num_img):
            im_input = Ft.to_pil_image(i_input[i, :, :, :].data.cpu())
            im_target = Ft.to_pil_image(target[i, :, :, :].data.cpu())
            input_blur = Ft.to_tensor(im_input.filter(ImageFilter.GaussianBlur())).cuda()
            target_blur = Ft.to_tensor(im_target.filter(ImageFilter.GaussianBlur())).cuda()
            color_loss +=  self.criterion(input_blur, target_blur)
        return color_loss/num_img
    
    def compute_content_loss(self, i_input, target):
        input_feats = self.model(i_input)
        target_feats = self.model(target)
        nr_feats = len(input_feats)
        content_loss = 0
        for i in range(nr_feats):
            content_loss += self.criterion(input_feats[i], target_feats[i]).item()
        return content_loss/nr_feats

    def compute_style_loss(self, i_input, target):
        input_feats = self.model(i_input)
        target_feats = self.model(target)
        nr_feats = len(input_feats)
        style_loss = 0
        for i in range(nr_feats):
            gi = gram_matrix(input_feats[i])
            gt = gram_matrix(target_feats[i])
            style_loss += self.criterion(gt, gi).item()
        return style_loss/nr_feats
    
    def compute_perceptual_loss(self, synthetic, real):
        color_loss = self.compute_color_loss(synthetic.detach(), real)
        style_loss = self.compute_style_loss(synthetic.detach(), real)
        content_loss = self.compute_content_loss(synthetic.detach(), real)
        perceptual_loss = color_loss + 10 * style_loss + content_loss
        return perceptual_loss

