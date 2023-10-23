from trainpipemodels import *
import torch.nn as nn
import torch
from data import NoduleData
from torchvision.transforms import functional as TF

dataset = NoduleData("finalCXRDataset/final.csv", "finalCXRDataset/images", 140)

im = TF.to_pil_image(dataset[2][0])
im.show()