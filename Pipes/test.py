import numpy as np
import torch
from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from utils import place_nodules, get_centerx_getcentery, get_dim, get_fake_difficulties

import matplotlib.pyplot as plt

# CUDA DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------- ACTUAL CODE ---------------------------------------------
MAX_SIZE = 140
NUM = 100

# Generators
lung_generator = LoadLungGenerator(device=device, path="OBGAN2/savedmodels/080000_g.model")
nodule_generator = LoadNoduleGenerator(device=device, path="OBGAN2/savedmodels/nodulegenerator.pth")

lengths = np.linspace(0, 140, num=70)
predicted_lengths = []

for length in lengths:
    diff = length * length/MAX_SIZE

    nodules = nodule_generator.nodule_predict(diff=diff, num_images=NUM)

    sum_width, sum_height = 0

    for nodule in nodules:
        w, h = get_dim(nodule=nodule, max_size=MAX_SIZE)
        sum_width += w
        sum_height += h
    
    avg_width, avg_height = sum_width/len(nodules), sum_height/len(nodules)

    predicted_lengths.append(avg_width)


plt.plot(lengths, predicted_lengths)
plt.show()