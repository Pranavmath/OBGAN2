from data import OBData
from PIL import Image

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

data = ob_dataset.get_from_difficulty(0.51, 0.085)

i = 0

nodule_imgs = {}

for image, nodules in data:
    for nodule in nodules:
        xmin, ymin, xmax, ymax = nodule
        centerx, centery = (xmin + xmax)//2 , (ymin + ymax)//2

        nodule_img = image.crop([centerx - 70, centery - 70, centerx + 70, centery + 70])
        
        nodule_img1 = nodule_img.transpose(Image.FLIP_LEFT_RIGHT)
        nodule_img2 = nodule_img.transpose(Image.FLIP_TOP_BOTTOM)
        nodule_img3 = nodule_img.transpose(Image.ROTATE_90)
        nodule_img4 = nodule_img.transpose(Image.ROTATE_180)
        nodule_img5 = nodule_img.transpose(Image.ROTATE_270)
        
        nodule_img1.save(f"nodulegendataset/trainB/{i}.jpg")
        nodule_imgs[i] = (xmax - xmin) * (ymax - ymin) / (140 ** 2)
        i += 1

        nodule_img2.save(f"nodulegendataset/trainB/{i}.jpg")
        nodule_imgs[i] = (xmax - xmin) * (ymax - ymin) / (140 ** 2)
        i += 1

        nodule_img3.save(f"nodulegendataset/trainB/{i}.jpg")
        nodule_imgs[i] = (xmax - xmin) * (ymax - ymin) / (140 ** 2)
        i += 1

        nodule_img4.save(f"nodulegendataset/trainB/{i}.jpg")
        nodule_imgs[i] = (xmax - xmin) * (ymax - ymin) / (140 ** 2)
        i += 1

        nodule_img5.save(f"nodulegendataset/trainB/{i}.jpg")
        nodule_imgs[i] = (xmax - xmin) * (ymax - ymin) / (140 ** 2)
        i += 1


import json

with open('data.json', 'w') as fp:
    json.dump(nodule_imgs, fp)