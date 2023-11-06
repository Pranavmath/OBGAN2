import numpy as np
import torch
from torch import optim
from mmdet.apis import init_detector
from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
from data import OBData
from utils import place_nodules, get_centerx_getcentery, get_dim, get_fake_difficulties
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ob_dataset = OBData(csv="OBGAN2/finalCXRDataset/final.csv", img_dir="OBGAN2/finalCXRDataset/images", control_img_dir="OBGAN2/finalCXRDataset/controlimages")

# Specify the path to model config and checkpoint file
config_file = 'OBGAN2/savedcvmodelmmdet/config.py'
checkpoint_file = 'OBGAN2/savedcvmodelmmdet/dino-4scale_r50_8xb2-12e_coco.pth'

# Build the model from a config file and a checkpoint file
cv_model = LoadCVModel(model=init_detector(config_file, checkpoint_file, device='cuda:0'), device=device)

# Optimizer
# Lr needs to be low enough or error
optimizer = optim.Adam(params=cv_model.model.parameters(), lr=0.0007)

# Set to training mode (if not already)
cv_model.model.train()


all_images_bboxes = ob_dataset.all_above_difficulty(0) + ob_dataset.get_control_images(num=1493)

NUM_EPOCHS = 33

print("Starting")

for _ in range(NUM_EPOCHS):
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
  
  print(f"Epoch: {e}, Avg Loss: {avg_loss}")


# Saving the model
torch.save(cv_model.model.state_dict(), "dinonormal.pth")
