import numpy as np
import torch
from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from utils import place_nodules, get_centerx_getcentery, get_dim
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

diffs = np.linspace(0, 1, num=500)
avg_ious = []

for diff in tqdm(diffs):
    sum_iou = 0

    for _ in range(5):
        # Number of nodules in this given fake image
        num_nodules_in_image = random.randint(1, 4)

        # Generates the background lung image on which the nodules will be placed on
        background_lung_image = lung_generator.lung_predict(num_images=1)

        # Generates the nodules (at the curr diff) to be placed on the backgroudn lung image (always MAX_SIZE x MAX_SIZE and have a black border around them)
        nodules = nodule_generator.nodule_predict(diff=diff, num_images=num_nodules_in_image)
        
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

        # ----------------------------
                
        iou = eval(image=fake_image, gt_bboxes=fake_bboxes)

        sum_iou += iou
    

    avg_iou = sum_iou/5
    avg_ious.append(avg_iou)


plt.plot(diffs, avg_ious)
plt.savefig("taquavion.png")