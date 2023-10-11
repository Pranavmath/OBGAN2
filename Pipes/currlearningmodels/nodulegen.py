# FROM _____ import NODULE_GEN

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from PIL import Image
from math import sqrt


def get_nodule_generator(device, path):
    #nodule_gen = NODULE_GEN(**insert_kwargs).to(device)
    # Do this by uploading the .model file dont use github
    nodule_gen.load_state_dict(torch.load(path, map_location=device))
    return nodule_gen

# Return pil image(s)
def nodule_predict(generator, diff, num_images, device):
  imgs = []
  for _ in range(num_images):
    #input_code_size = ______

    with torch.no_grad():
      images = generator(torch.randn(num_images, input_code_size).to(device), # Passing in difficulty using diff 
                        ).data.cpu()

    grid = make_grid(images, normalize=True, range=(-1, 1))
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    imgs.append(im)
  
  return imgs