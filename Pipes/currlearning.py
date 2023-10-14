import numpy as np
import torch
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

from Pipes.currlearningmodels.lunggen import get_lung_generator, lung_predict
from Pipes.currlearningmodels.nodulegen import get_nodule_generator, nodule_predict
from Pipes.currlearningmodels.mmdetmodel import predict_cv
from data import OBData
from utils import place_nodules, get_centerx_getcentery
import random

# The maximum size of a nodule (length)
MAX_SIZE = 140
# Step in difficulty when training the model (always negative beacause smaller difficulty means it  harder)
STEP = -0.1
# Used to get all the real images that have a difficulty that it at max delta away from the current step in difficlty
DELTA_DIFF = 0.05
# Ending Difficulty
END_DIFF = 0.01
# Number of Fake Images
NUM_FAKE = 100
# CUDA DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Things to keep note of: 
    1. Higher difficulty is easier and vice verse
    2. Min/Max Nodule Size (Generated): 0x0 - 140x140 
    3. Min/Max Nodule Difficulty (Generated): 0 - 1
    4. Number of nodules in Each (real) image: 1-4
"""

# ------------------------------------- ACTUAL CODE ---------------------------------------------

ob_dataset = OBData()


# Specify the path to model config and checkpoint file
config_file = 'cvmodelmmdet/config.py'
checkpoint_file = 'cvmodelmmdet/_____INSERT_PRETRAINED_MODEL_SAVE___FILE'

# Build the model from a config file and a checkpoint file
# Returns a nn.Module
cv_model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Generators
lung_generator = get_lung_generator()

nodule_generator = get_nodule_generator()


# Need to get real bboxes and fake bboxes

# Current (Start) Difficulty
curr_diff = 1

while curr_diff >= END_DIFF:
    real_images = ob_dataset.get_from_difficulty(curr_diff, DELTA_DIFF)
    fake_images = []

    for _ in range(NUM_FAKE):
        num_nodules_in_image = random.randint(1, 4)

        background_lung_image = lung_predict(lung_generator, num_images=1, device=device)
        nodules = nodule_predict(nodule_generator, diff=curr_diff, num_images=num_nodules_in_image, device=device)
        center_xys = get_centerx_getcentery(num_nodules=num_nodules_in_image)

        fake_image = place_nodules(background_image=background_lung_image, nodules=nodules, center_xy_nodules=center_xys)
        fake_images.append(fake_image)
    
    all_images = random.shuffle(real_images + fake_images)

    for image in all_images:
        loss = predict_cv(cv_model, image, )
