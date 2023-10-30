import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from mmdet.apis import init_detector

from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
from data import OBData
from utils import place_nodules, get_centerx_getcentery, get_dim, get_fake_difficulties
import random

# The maximum size of a nodule (length)
MAX_SIZE = 140
# Starting Difficulty
START_DIFF = 0.96
# Step in difficulty when training the model (change in difficulty must be negative beacause smaller difficulty means it's harder)
STEP = 0.05
# Ending Difficulty
END_DIFF = 0.01
# Number of Fake Images at the starting difficulty
START_NUM_FAKE = 100
# Number of Fake Images at the ending difficulty
END_NUM_FAKE = 300
# Number of epochs for each step in difficulty
NUM_EPOCHS_FOR_STEP = 3
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

writer = SummaryWriter()


ob_dataset = OBData(csv="OBGAN2/finalCXRDataset/final.csv", img_dir="OBGAN2/finalCXRDataset/images", control_img_dir="OBGAN2/finalCXRDataset/controlimages")


# Specify the path to model config and checkpoint file
config_file = 'OBGAN2/savedcvmodelmmdet/config.py'
checkpoint_file = 'OBGAN2/savedcvmodelmmdet/dino-4scale_r50_8xb2-12e_coco.pth'


# Build the model from a config file and a checkpoint file
cv_model = LoadCVModel(model=init_detector(config_file, checkpoint_file, device='cuda:0'), device=device)


# Generators
lung_generator = LoadLungGenerator(device=device, path="OBGAN2/savedmodels/080000_g.model")
nodule_generator = LoadNoduleGenerator(device=device, path="OBGAN2/savedmodels/nodulegenerator.pth")


# Optimizer
# Lr needs to be low enough or error
optimizer = optim.Adam(params=cv_model.model.parameters(), lr=0.0007)


# Set to training mode (if not already)
cv_model.model.train()


# Current (Start) Difficulty
curr_diff = START_DIFF

print("Starting")

while curr_diff >= END_DIFF:
    # Num fake images (calculated from curr_diff)
    num_fake = int(END_NUM_FAKE + ((START_NUM_FAKE - END_NUM_FAKE)/(START_DIFF - END_DIFF)) * (curr_diff - END_DIFF))

    for e in range(NUM_EPOCHS_FOR_STEP):
        # Gets all the real images and nodules - [(real_image1, real_bbox1), ...] above a given difficulty
        real_images_bboxes = ob_dataset.all_above_difficulty(curr_diff)

        # All the fake images and nodules - [(fake_image1, fake_bbox1), ...] at (and above) this difficulty
        fake_images_bboxes = []

        # The fake difficulties to use (a bunch of random difficulties at the current difficulty and above)
        fake_difficulties = get_fake_difficulties(curr_difficulty=curr_diff, num_of_difficulties=num_fake)

        # Gets NUM_FAKE number of fake images/bboxes at the sampled fake difficulties
        for fake_diff in fake_difficulties:
            # Number of nodules in this given fake image
            num_nodules_in_image = random.randint(1, 4)

            # Generates the background lung image on which the nodules will be placed on
            background_lung_image = lung_generator.lung_predict(num_images=1)

            # Generates the nodules (at the curr diff) to be placed on the backgroudn lung image (always MAX_SIZE x MAX_SIZE and have a black border around them)
            nodules = nodule_generator.nodule_predict(diff=fake_diff, num_images=num_nodules_in_image)
            
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
        

        # Get the control images (no nodules) and bboxes. Make sure to get the same amount as the real images + fake images so data is balanced
        control_images_bboxes = ob_dataset.get_control_images(num=len(real_images_bboxes) + len(fake_images_bboxes))

        # Shuffles the real (with nodule), fake, and control images/bboxes
        all_images_bboxes = real_images_bboxes + fake_images_bboxes + control_images_bboxes
        random.shuffle(all_images_bboxes)

        
        sum_loss = 0
        
        # Trains cv model on all images
        for image, bbox in all_images_bboxes:
            # Clear all gradients
            optimizer.zero_grad()

            # Label of each object in img (always 0 because there are only nodules)
            labels = [0 for _ in range(len(bbox))]

            # Takes image, bboxes of objects, and labels of objcets and gets loss as dict
            loss = cv_model.predict_cv(img=image, gt_bboxes=bbox, gt_labels=labels)

            # Do the loss backwards on 3 components: loss_cls, loss_bbox, loss_iou
            total_loss = loss["loss_cls"] + loss["loss_bbox"]
            total_loss.backward()
            #loss["loss_iou"].backward()

            optimizer.step()

            sum_loss += total_loss.item()
        

        avg_loss = sum_loss/len(all_images_bboxes)
        
        writer.add_scalar("Loss/train", avg_loss, curr_diff - STEP * (e/NUM_EPOCHS_FOR_STEP))

    

    # Steps the current difficulty down (makes it harder)
    curr_diff -= STEP


# Saving the model
torch.save(cv_model.model.state_dict(), "dino.pth")