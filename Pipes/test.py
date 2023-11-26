import numpy as np
import torch
from data import OBData

from tqdm import tqdm

import matplotlib.pyplot as plt

# CUDA DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

# ------------------------------------- ACTUAL CODE ---------------------------------------------
# The maximum size of a nodule (length)
MAX_SIZE = 140
# Starting Difficulty
START_DIFF = 0.4275
# Step in difficulty when training the model (change in difficulty must be negative beacause smaller difficulty means it's harder)
STEP = 0.0167
# Ending Difficulty
END_DIFF = 0.01
# Number of Fake Images at the starting difficulty
START_NUM_FAKE = 400
# Number of Fake Images at the ending difficulty
END_NUM_FAKE = 500
# Number of epochs for each step in difficulty
NUM_EPOCHS_FOR_STEP = 2
# Batch Size
BATCH_SIZE = 10

# Current (Start) Difficulty
curr_diff = START_DIFF

print(len(ob_dataset.all_above_difficulty(0)))