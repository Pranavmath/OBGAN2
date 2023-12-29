import numpy as np
import torch
from torch import optim

from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
from data import OBData
from utils import get_centerx_getcentery, get_fake_difficulties, batch_data, get_width_and_height, get_mask_image_patch
import random
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

valid_images_bboxes = valid_dataset.all_above_difficulty(0) + valid_dataset.get_control_images(num=83)

# ---------------------------------------------------------------------------------------------

# Build the model from a config file and a checkpoint file
cv_model = LoadCVModel(device=device)

# Generators
nodule_generator = LoadNoduleGenerator(device=device, path="INSERT PATH HERE")

# Optimizer
# Lr needs to be low enough or error
optimizer = optim.Adam(params=cv_model.model.parameters(), lr=0.0007)


# Set to training mode (if not already)
cv_model.model.train()

wandb.watch(cv_model.model, log_freq=100)

# Current (Start) Difficulty
curr_diff = START_DIFF

print("Starting")

while curr_diff >= END_DIFF:
    # Num fake images (calculated from curr_diff)
    num_fake = int(END_NUM_FAKE + ((START_NUM_FAKE - END_NUM_FAKE)/(START_DIFF - END_DIFF)) * (curr_diff - END_DIFF))
    
    # Gets all the real images and nodules - [(real_image1, real_bbox1), ...] above a given difficulty
    real_images_bboxes = ob_dataset.all_above_difficulty(curr_diff)
    
    if curr_diff <= 0.07:
        num_epochs = 4
    else:
        num_epochs = NUM_EPOCHS_FOR_STEP
    
    for e in range(num_epochs):
        # All the fake images and nodules - [(fake_image1, fake_bbox1), ...] at (and above) this difficulty
        fake_images_bboxes = []

        # The fake difficulties to use (a bunch of random difficulties at the current difficulty and above)
        fake_difficulties = get_fake_difficulties(curr_difficulty=curr_diff, num_of_difficulties=num_fake)

         # Gets the background lung image on which the nodules will be placed on
        background_lung_images = [t[0] for t in ob_dataset.get_control_images(num=num_fake)]

        
        # Gets NUM_FAKE number of fake images/bboxes at the sampled fake difficulties
        for i in range(len(background_lung_images)):
            fake_diff = fake_difficulties[i]
            background_lung_image = background_lung_images[i]

            # Number of nodules in this given fake image
            num_nodules_in_image = random.randint(1, 3)
            
            # Generates x and y values to put the center of each nodule on the lung image (using a distrubution of nodules in real images) 
            center_xys = get_centerx_getcentery(num_nodules=num_nodules_in_image)
            
            # Sizes of each nodule: [(width1, height1), ...]
            sizes_nodules = [get_width_and_height(diff=fake_diff) for _ in range(num_nodules_in_image)]

            # Gets the bbox of each nodule on the lung image
            fake_bboxes = []
            for nodule_idx in range(num_nodules_in_image):
                centerx, centery = center_xys[nodule_idx]
                width, height = sizes_nodules[nodule_idx]
                width, height = int(width), int(height)
                
                # Lung patch and mask
                lung_patch, mask = get_mask_image_patch(lung_img=background_lung_image, centerx=centerx, centery=centery, width=width, height=height)
                
                # Length of lung patch
                length_lung_patch = lung_patch.size[0]

                # Lung Patch with Nodule
                nodule_patch = nodule_generator.nodule_predict(masks=[mask], lung_patches=[lung_patch])[0].resize((length_lung_patch, length_lung_patch))

                # Adds Lung Patch with Nodule to background lung image
                background_lung_image.paste(nodule_patch, box=[centerx - length_lung_patch // 2, centery - length_lung_patch // 2, centerx + length_lung_patch // 2, centery + length_lung_patch // 2])

                # Format: xmin, ymin, xmax, ymax
                bbox = [centerx - width//2, centery - height//2, centerx + width//2, centery + height//2]
                fake_bboxes.append(bbox)
            

            fake_images_bboxes.append((background_lung_image, fake_bboxes))
        
        
        # Get the control images (no nodules) and bboxes. Make sure to get the same amount as the real images + fake images so data is balanced
        control_images_bboxes = ob_dataset.get_control_images(num=len(real_images_bboxes) + len(fake_images_bboxes))

        # Shuffles the real (with nodule), fake, and control images/bboxes
        all_images_bboxes = real_images_bboxes + control_images_bboxes + fake_images_bboxes 
        random.shuffle(all_images_bboxes)

        all_images_bboxes = batch_data(all_images_bboxes, BATCH_SIZE)


        sum_train_loss = 0

        # Trains cv model on all images
        for images, batch_bboxes in all_images_bboxes:
            # Clear all gradients
            optimizer.zero_grad()

            # Takes image, bboxes of objects of objcets and gets loss as dict
            loss_dict = cv_model.predict_cv(images=images, batch_bboxes=batch_bboxes)

            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            sum_train_loss += losses.item()
        

        avg_train_loss = sum_train_loss/len(all_images_bboxes)

        wandb.log({"train epoch avg loss": avg_train_loss})



    random.shuffle(valid_images_bboxes)
    sum_valid_iou = 0

    # Trains cv model on all images
    for image, gt_bboxes in valid_images_bboxes:
        # Takes image, bboxes of objects of objcets and gets loss as dict
        with torch.no_grad():
            iou = cv_model.iou(image=image, gt_bboxes=gt_bboxes)
            sum_valid_iou += iou["iou"].item()


    avg_valid_iou = sum_valid_iou/len(valid_images_bboxes)

    if (avg_valid_iou >= 0.36):
        torch.save(cv_model.model.state_dict(), f"dino{avg_valid_iou}.pth")

    wandb.log({"valid diff step avg iou": avg_valid_iou})



    # Steps the current difficulty down (makes it harder)
    curr_diff -= STEP


# Saving the model
torch.save(cv_model.model.state_dict(), "dino.pth")
