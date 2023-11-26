import numpy as np
import torch
from torch import optim

from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
from data import OBData
from utils import batch_data
import random
from time import time

import wandb

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
# CUDA DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Things to keep note of: 
    1. Higher difficulty is easier and vice verse
    2. Min/Max Nodule Size (Generated): 0x0 - 140x140 
    3. Min/Max Nodule Difficulty (Generated): 0 - 1
    4. Number of nodules in Each (real) image: 1-4
    5. Right now we dont train the model on easier images compared to that difficulty (higher difficulty)
"""

# ------------------------------------- ACTUAL CODE ---------------------------------------------

wandb.init(project="curr_learning")

ob_dataset = OBData(csv="OBGAN2/finalCXRDataset/final.csv", img_dir="OBGAN2/finalCXRDataset/images", control_img_dir="OBGAN2/finalCXRDataset/controlimages")

valid_dataset = OBData(csv="OBGAN2/newtest/newtest.csv", img_dir="OBGAN2/newtest/images", control_img_dir="OBGAN2/newtest/controlimages")

train_images_bboxes = ob_dataset.all_above_difficulty(0) + ob_dataset.get_control_images(num=1493)
valid_images_bboxes = valid_dataset.all_above_difficulty(0) + valid_dataset.get_control_images(num=83)

# ---------------------------------------------------------------------------------------------

# Build the model from a config file and a checkpoint file
cv_model = LoadCVModel(device=device)


# Optimizer
# Lr needs to be low enough or error
optimizer = optim.Adam(params=cv_model.model.parameters(), lr=0.0007)


# Set to training mode (if not already)
cv_model.model.train()

wandb.watch(cv_model.model, log_freq=100)

print("Starting")

for _ in range(50):
  random.shuffle(train_images_bboxes)
  train_batched_images_bboxes = batch_data(train_images_bboxes, BATCH_SIZE)

  sum_train_loss = 0

  # Trains cv model on all images
  for images, batch_bboxes in train_batched_images_bboxes:
    # Clear all gradients
    optimizer.zero_grad()

    # Takes image, bboxes of objects of objcets and gets loss as dict
    loss_dict = cv_model.predict_cv(images=images, batch_bboxes=batch_bboxes)

    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()

    sum_train_loss += losses.item()

  avg_train_loss = sum_train_loss/len(train_batched_images_bboxes)

  wandb.log({"train epoch avg loss": avg_train_loss})



  random.shuffle(valid_images_bboxes)
  valid_batched_images_bboxes = batch_data(valid_images_bboxes, BATCH_SIZE)

  sum_valid_loss = 0

  # Trains cv model on all images
  for images, batch_bboxes in valid_batched_images_bboxes:
    # Takes image, bboxes of objects of objcets and gets loss as dict
    with torch.no_grad():
        loss_dict = cv_model.predict_cv(images=images, batch_bboxes=batch_bboxes)
        losses = sum(loss for loss in loss_dict.values())
        sum_valid_loss += losses.item()


  avg_valid_loss = sum_valid_loss/len(valid_batched_images_bboxes)

  wandb.log({"valid diff step avg loss": avg_valid_loss})


# Saving the model
torch.save(cv_model.model.state_dict(), "dino.pth")
