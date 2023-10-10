from typing import Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
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
    
    def __len__(self):
        # Num of images in img_dir
        return len(os.listdir(self.img_dir))
    
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
    
    def __getitem__(self, index):
        image_name, nodule_bbox = self.nodules[index]
        image_path = os.path.join(self.img_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        nodule = image.crop(nodule_bbox)

        width, height = nodule.size

        if nodule.size[0] > self.max_size:
            nodule = nodule.crop((0, (height-self.max_size)//2, width, self.max_size + ((height-self.max_size)//2)))
        if nodule.size[1] > self.max_size:
            nodule = nodule.crop(((width - self.max_size)//2, 0, self.max_size + ((width-self.max_size)//2), height))

        width, height = nodule.size
        background = Image.new("RGB", (self.max_size, self.max_size))
        background.paste(nodule, ((self.max_size - width)//2, (self.max_size - height)//2))

        #background = self.to_tensor(background)
        
        image_area = image.size[0] * image.size[1]
        diff = nodule_difficulty(nodule_bbox, image_area, (self.max_size)**2)

        return background, diff

