import numpy as np
import torch
from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from utils import place_nodules, get_centerx_getcentery, get_dim
from data import OBData
import random
from tqdm import tqdm
import torchvision.transforms.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import torchmetrics

# --------------------------------------------------------------------------

MAX_SIZE = 140

# CUDA DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

metric = torchmetrics.detection.mean_ap.MeanAveragePrecision()

curr_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
in_features = curr_model.roi_heads.box_predictor.cls_score.in_features
# define a new head for the detector with required number of classes
curr_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

curr_model.load_state_dict(torch.load("/content/rcnn_curr.pth"))
curr_model.to(device)
curr_model.eval()

valid_dataset = OBData(csv="OBGAN2/newtest/newtest.csv", img_dir="OBGAN2/newtest/images", control_img_dir="OBGAN2/newtest/controlimages")

# ------------------------------------- ACTUAL CODE ---------------------------------------------

def eval(image, gt_bboxes):
    prediction = curr_model([F.to_tensor(image).to(device)])[0]

    # Bounding boxes
    bounding_boxes = prediction["boxes"].cpu()
    confidence_score = prediction["scores"].cpu()

    pred = {
        "boxes": torch.tensor(bounding_boxes),
        "scores": torch.tensor(confidence_score),
        "labels": torch.ones(len(bounding_boxes), dtype=torch.int8)
    }

    t = {
        "boxes": torch.tensor(gt_bboxes),
        "labels":  torch.ones(len(gt_bboxes), dtype=torch.int8)
    }

    iou = metric([pred], [t])

    return iou


# Generators
lung_generator = LoadLungGenerator(device=device, path="OBGAN2/savedmodels/080000_g.model")
nodule_generator = LoadNoduleGenerator(device=device, path="OBGAN2/savedmodels/nodulegenerator.pth")

diffs = np.linspace(0, 1, num=100)
avg_ious = []

for diff in tqdm(diffs):
    sum_iou = 0

    ayo = valid_dataset.get_from_difficulty(difficulty=diff, delta=0.02)

    for image, bboxes in ayo:    
        iou = eval(image=image, gt_bboxes=bboxes)
        sum_iou += iou
    

    avg_iou = sum_iou/len(ayo)
    avg_ious.append(avg_iou)


plt.plot(diffs, avg_ious)
plt.savefig("taquavion.png")