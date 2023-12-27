from data import OBData
from PIL import Image, ImageDraw
from tqdm import tqdm

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

data = ob_dataset.all_below_difficulty(0.51)

i = 0

nodule_imgs = {}

def overlay_black(img, bbox):
    img1 = ImageDraw.Draw(img)
    img1.rectangle(bbox, fill="#000000")
    return img

for image, nodules in tqdm(data):
    for nodule in nodules:
        xmin, ymin, xmax, ymax = nodule

        nodule_img = image.crop([xmin - 7, ymin - 7, xmax + 7, ymax + 7])
        nodule_img = nodule_img.resize((140, 140))

        nodule_img.save(f"nodulegendataset/trainB/{i}.jpg")
        i += 1

"""
import json

with open('data.json', 'w') as fp:
    json.dump(nodule_imgs, fp)
"""