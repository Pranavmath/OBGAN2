from typing import Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import os

def collate_fn(batch):
    return tuple(zip(*batch))


def nodule_dict(df):
    # xmin, ymin, xmax, ymax format
    nodules = {}
    for i in range(len(df.index)):
        image_name = df["image"][i]
        bbox = (df["xmin"][i], df["ymin"][i], df["xmax"][i], df["ymax"][i])

        if image_name in nodules.keys():
            nodules[image_name].append(bbox)
        else:
            nodules[image_name] = [bbox]
    
    return nodules


# Returns the percent of the area of the image taken by the biggest nodule in the image
# Higher means easier, Lower means harder
# Expects bboxes of each nodule
def image_difficulty(image, nodules):
    _, width, height = image.size()
    area = width * height

    return max([((nodule[2]-nodule[0]) * (nodule[3]-nodule[1]))/area for nodule in nodules]) * 100


# Returns the percent of the area of the image taken by the nodule in the image
# Higher means easier, Lower means harder 
# Min - 0, max - 1
def nodule_difficulty(nodule_bbox, image_area, max_area):
    nodule_area = ((nodule_bbox[2]-nodule_bbox[0]) * (nodule_bbox[3]-nodule_bbox[1]))
    scaler = image_area/max_area
    
    if nodule_area > max_area:
        return (max_area/image_area) * scaler
    else:
        return (nodule_area/image_area) * scaler
 
 
# Dataset of lung images, bboxes, and its difficulties
class OBData(Dataset):
    def __init__(self, csv, img_dir, image_transforms=None):
        self.csv = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transforms = image_transforms

        if set(self.csv.keys()) != set(["image", "xmin", "ymin", "xmax", "ymax", "label"]):
            raise Exception("Wrong csv format") 
        
        file_name = os.listdir(self.img_dir)
        for i in range(len(self.csv.index)):
            if not (self.csv["image"][i] in file_name):
                raise FileNotFoundError("File in CSV not found in img_dir folder")
        
        self.nodule_dict = nodule_dict(self.csv)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        
        # image file name - Its difficulty
        self.difficulties = {}
        for file_name in os.listdir(self.img_dir):
            image_path = os.path.join(self.img_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            image = self.to_tensor(image)
            image = self.to_tensor(image)

            nodules = self.nodule_dict[file_name]
            nodules = torch.tensor(nodules)

            self.difficulties[file_name] = image_difficulty(image, nodules)

    
    def __len__(self):
        # Num of images in img_dir
        return len(os.listdir(self.img_dir))
    
    # Return list = [(image1, bbox1) ...] of images (PIL not Tensor) close to a given difficulty
    def get_from_difficulty(self, difficulty, delta):
        image_names = [pair[0] for pair in self.difficulties.items() if abs(pair[1]-difficulty) <= delta]
        return [(Image.open(os.path.join(self.img_dir, image_name)), self.nodule_dict[image_name]) for image_name in image_names]
    
    # Returns image and nodules as tensor along with its difficulty
    def __getitem__(self, index):
        image_name = os.listdir(self.img_dir)[index]
        image_path = os.path.join(self.img_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image = self.to_tensor(image)
        if self.transforms:
            image = self.transforms(image)

        nodules = self.nodule_dict[image_name]
        nodules = torch.tensor(nodules)
        
        diff = image_difficulty(image, nodules)

        return image, nodules, diff


# Dataset of nodules and its difficulties
class NoduleData(Dataset):
    def __init__(self, csv, img_dir, max_size):
        self.csv = pd.read_csv(csv)
        self.img_dir = img_dir
        self.max_size = max_size

        if set(self.csv.keys()) != set(["image", "xmin", "ymin", "xmax", "ymax", "label"]):
            raise Exception("Wrong csv format") 
        
        file_name = os.listdir(self.img_dir)
        for i in range(len(self.csv.index)):
            if not (self.csv["image"][i] in file_name):
                raise FileNotFoundError("File in CSV not found in img_dir folder")

        self.nodule_dict = nodule_dict(self.csv)

        self.nodules = []
        for image_name, nodules in self.nodule_dict.items():
            for nodule in nodules:
                self.nodules.append((image_name, nodule))
        
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        # Mask to make nodules ellptical
        size = (self.max_size, self.max_size)
        self.mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(self.mask) 
        draw.ellipse((0, 0) + size, fill=255)
        self.mask = self.mask.convert("RGB")

    def __len__(self):
        return len(self.nodules)
    
    # Return a nodule (as a tensor) and its difficulty
    def __getitem__(self, index):
        image_name, nodule_bbox = self.nodules[index]
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        nodule = image.crop(nodule_bbox)

        width, height = nodule.size
        
        # Making sure that the width and height of the nodule is less than the max size
        if nodule.size[0] > self.max_size:
            nodule = nodule.crop((0, (height-self.max_size)//2, width, self.max_size + ((height-self.max_size)//2)))
        if nodule.size[1] > self.max_size:
            nodule = nodule.crop(((width - self.max_size)//2, 0, self.max_size + ((width-self.max_size)//2), height))

        width, height = nodule.size

        
        # Making the nodule elliptical (hadamard product of mask with the image)
        mask = np.array(self.mask.resize((width, height))) / 255 
        nodule = Image.fromarray(np.multiply(np.array(nodule), mask.astype(np.uint8)))
        

        # Adding the nodule to a maxsize x maxsize black background       
        background = Image.new("RGB", (self.max_size, self.max_size))
        background.paste(nodule, ((self.max_size - width)//2, (self.max_size - height)//2))

        background = self.to_tensor(background)
        
        image_area = image.size[0] * image.size[1]
        diff = nodule_difficulty(nodule_bbox, image_area, (self.max_size)**2)

        return background, diff

