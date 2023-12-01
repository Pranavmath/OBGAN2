from data import OBData
from PIL import Image
from utils import get_centerx_getcentery

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

data = ob_dataset.get_from_difficulty(0.51, 0.085)


nodule_imgs = []
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
        
        nodule_imgs.append(nodule_img)
        nodule_imgs.append(nodule_img1)
        nodule_imgs.append(nodule_img2)
        nodule_imgs.append(nodule_img3)
        nodule_imgs.append(nodule_img4)
        nodule_imgs.append(nodule_img5)


control_data = ob_dataset.get_control_images(num = len(nodule_imgs))

patch_imgs = []
for image, _ in control_data:
    centerx, centery = get_centerx_getcentery(1)[0]
    patch_img = image.crop([centerx - 70, centery - 70, centerx + 70, centery + 70])
    patch_imgs.append(patch_img)


for i, img in enumerate(patch_imgs):
    img.save(f"nodulegendataset/trainA/{i}.jpg")

for i, img in enumerate(nodule_imgs):
    img.save(f"nodulegendataset/trainB/{i}.jpg")