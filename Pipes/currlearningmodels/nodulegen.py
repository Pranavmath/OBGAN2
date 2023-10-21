# FROM _____ import NODULE_GEN as BASE_GEN

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
        #self.nodule_gen = NODULE_GEN(**insert_kwargs).to(device)
        # Do this by uploading the .model file dont use github
        self.nodule_gen.load_state_dict(torch.load(path, map_location=device))

    # Return pil image(s) fron nodule generator
    def nodule_predict(self, diff, num_images):
        """
        Takes in difficulty and num_images (both int) and returns a list of PIL Images
        """
        #input_code_size = ______
        imgs = []
        
        for _ in range(num_images):
          with torch.no_grad():
            images = self.nodule_gen(torch.randn(1, input_code_size).to(self.device), # Passing in difficulty using diff 
                              ).data.cpu()
          
          grid = make_grid(images, normalize=True, range=(-1, 1))
          # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
          ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
          im = Image.fromarray(ndarr)
          
          imgs.append(im)
        
        return imgs



# -------------------------------- CLASSES BELLOW FOR GENERATOR LOADING (DONT CHANGE) --------------------
