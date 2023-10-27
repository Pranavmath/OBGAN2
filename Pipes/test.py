from Pipes.currlearningmodels.lunggen import LoadLungGenerator
from Pipes.currlearningmodels.nodulegen import LoadNoduleGenerator
from Pipes.currlearningmodels.mmdetmodel import LoadCVModel
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generators
lung_generator = LoadLungGenerator(device=device, path="OBGAN2/savedmodels/080000_g.model")
nodule_generator = LoadNoduleGenerator(device=device, path="OBGAN2/savedmodels/nodulegenerator.pth")

lung_generator.lung_predict(1).save("lung.png")
nodule_generator.nodule_predict(0.5, 2)[0].save("nodule.pngx")
