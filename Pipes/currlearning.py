import numpy as np
import torch
from torch import optim

from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
from data import OBData
from utils import place_nodules, get_centerx_getcentery, get_dim
import random

# The maximum size of a nodule (length)
MAX_SIZE = 140
# Step in difficulty when training the model (always negative beacause smaller difficulty means it  harder)
STEP = 0.1
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
config_file = 'savedcvmodelmmdet/config.py'
checkpoint_file = 'savedcvmodelmmdet/_____INSERT_PRETRAINED_MODEL_SAVE___FILE'


# Build the model from a config file and a checkpoint file
# Returns a nn.Module
cv_model = LoadCVModel(model=init_detector(config_file, checkpoint_file, device='cuda:0'), device=device)


# Generators
lung_generator = LoadLungGenerator()
nodule_generator = LoadNoduleGenerator()


# Optimizer
optimizer = optim.Adam(param=cv_model.model.parameters(), lr=0.007)


# Set to training mode (if not already)
cv_model.model.train()


# Current (Start) Difficulty
curr_diff = 1



while curr_diff >= END_DIFF:
    # Gets all the real images and nodules - [(real_image1, real_bbox1), ...] near a given difficulty
    real_images_bboxes = ob_dataset.get_from_difficulty(curr_diff, DELTA_DIFF)
    # All the fake images and nodules - [(fake_image1, fake_bbox1), ...] at this difficulty
    fake_images_bboxes = []

    # Gets NUM_FAKE number of fake images/bboxes
    for _ in range(NUM_FAKE):
        # Number of nodules in this given fake image
        num_nodules_in_image = random.randint(1, 4)

        # Generates the background lung image on which the nodules will be placed on
        background_lung_image = lung_generator.lung_predict(num_images=1)

        # Generates the nodules (at the curr diff) to be placed on the backgroudn lung image (always MAX_SIZE x MAX_SIZE and have a black border around them)
        nodules = nodule_generator.nodule_predict(diff=curr_diff, num_images=num_nodules_in_image)
        
        # Generates x and y values to put the center of each nodule on the lung image (using a distrubution of nodules in real images) 
        center_xys = get_centerx_getcentery(num_nodules=num_nodules_in_image)

        # Gets the bbox of each nodule on the lung image
        fake_bboxes = []
        for nodule_idx in range(len(nodules)):
            nodule = nodules[nodule_idx]
            centerx, centery = center_xys[nodule_idx]
            
            # Width, and height of nodule (need to do this since the generated nodule is placed on black background of MAX_SIZE x MAX_SIZE)
            nodule_width, nodule_height = get_dim(nodule=nodule, max_size=MAX_SIZE)
            
            # Format: xmin, ymin, xmax, ymax
            bbox = [centerx - nodule_width//2, centery - nodule_height//2, centerx + nodule_width//2, centery + nodule_height//2]
            fake_bboxes.append(bbox)
        
        # Places the center of each nodule on the generated lung iamge using the centerx and centerys
        fake_image = place_nodules(background_image=background_lung_image, nodules=nodules, center_xy_nodules=center_xys)

        fake_images_bboxes.append((fake_image, fake_bboxes))
    
    
    # Shuffles the real and fake images/bboxes
    all_images_bboxes = random.shuffle(real_images_bboxes + fake_images_bboxes)
    

    # Trains cv model on all images
    for image, bbox in all_images_bboxes:
        # Clear all gradients
        optimizer.zero_grad()

        # Label of each object in img (always 0 because there are only nodules)
        labels = [0 for _ in range(len(bbox))]

        # Takes image, bboxes of objects, and labels of objcets and gets loss as dict
        loss = cv_model.predict_cv(img=image, gt_bboxes=bbox, gt_labels=labels)

        # Do the loss backwards on 3 components: loss_cls, loss_bbox, loss_iou
        loss["loss_cls"].backward()
        loss["loss_bbox"].backward()
        loss["loss_iou"].backward()

        optimizer.step()
    

    curr_diff -= STEP
