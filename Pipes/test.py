from trainpipemodels import *
import torch.nn as nn
import torch

ng = NoduleGenerator(channels_noise=128, channels_img=3, features_g=32)
diff = DifficultyPredictor(channels_img=3)
dis = NoduleDiscriminator(channels_img=3)

noise = torch.rand(2, 128, 1, 1)
l = torch.rand(2, 1)
image = torch.rand(2, 3, 140, 140)

difficulty = diff(image)
realness = dis(image)
nodule = ng(noise, l)

print(difficulty.shape)
print(realness.shape)
print(nodule.shape)