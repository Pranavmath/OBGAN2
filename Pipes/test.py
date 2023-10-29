import torch.nn as nn
import torch
from torchvision.transforms import functional as TF
from currlearningmodels.nodulegen import LoadNoduleGenerator
from currlearningmodels.lunggen import LoadLungGenerator
from PIL import Image
import math

imgsize = (250, 250) #The size of the image

image = Image.new('RGB', imgsize) #Create the image

innerColor = [255, 255, 255] #Color at the center
outerColor = [20, 20, 20] #Color at the corners


for y in range(imgsize[1]):
    for x in range(imgsize[0]):

        #Find the distance to the center
        distanceToCenter = math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)

        #Make it on a scale from 0 to 1
        distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0]/2)

        #Calculate r, g, and b values
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)


        #Place the pixel        
        image.putpixel((x, y), (int(r), int(g), int(b)))

pad = 20
start_bright = 100

for y in range(imgsize[1]):
    for x in range(imgsize[0]):
        if (y - pad <= 0):
            brightness = (start_bright/pad) * y
            image.putpixel((x, y), (int(brightness), int(brightness), int(brightness)))
        
        if (y >= 250 - pad):
            brightness = (start_bright/pad) * (250-y)
            image.putpixel((x, y), (int(brightness), int(brightness), int(brightness)))
        
        if (x - pad <= 0):
            brightness = (start_bright/pad) * x
            image.putpixel((x, y), (int(brightness), int(brightness), int(brightness)))
        
        if (x >= 250 - pad):
            brightness = (start_bright/pad) * (250-x)
            image.putpixel((x, y), (int(brightness), int(brightness), int(brightness)))

image.save("mask.png")

# -----------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lung_generator = LoadLungGenerator(device=device, path="savedmodels/080000_g.model")
nodule_generator = LoadNoduleGenerator(device=device, path="savedmodels/nodulegenerator.pth")

nodule = nodule_generator.nodule_predict(diff=0.7, num_images=1)[0].convert("L")
background_lung_image = lung_generator.lung_predict(num_images=1).convert("L").resize((1024, 1024))
mask = image.resize(nodule.size).convert("L")

mask.show()

background_lung_image.paste(nodule, (300, 300), mask)
background_lung_image.show()