from trainpipemodels import NoduleGenerator

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from PIL import Image
from math import sqrt


class LoadNoduleGenerator():
    # Gets nodule generator
    def __init__(self, device, path):
        """
        Device (cuda string) and path to load Generator (String)
        """
        self.device = device

        # Values used for training
        self.channels_noise = 128
        self.channels_img = 3
        self.features_g = 16
        
        self.nodule_gen = NoduleGenerator(channels_noise=self.channels_noise, channels_img=self.channels_img, features_g=self.features_g).to(device)
        # Do this by uploading the .model file dont use github
        self.nodule_gen.load_state_dict(torch.load(path, map_location=device))

    # Return pil image(s) fron nodule generator
    def nodule_predict(self, diff, num_images):
        """
        Takes in difficulty and num_images (both int) and returns a list of PIL Images
        """
        imgs = []
        
        for _ in range(num_images):
          with torch.no_grad():
            image = self.nodule_gen(torch.randn(1, self.channels_noise, 1, 1).to(self.device), diff)[0]
          
          grid = make_grid(image, normalize=True, value_range=(-1, 1))
          # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
          ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
          im = Image.fromarray(ndarr)
          
          imgs.append(im)
        
        return imgs

