from currlearningmodels.nodulegen import LoadNoduleGenerator
from utils import get_mask_image_patch
from PIL import Image

device = "cuda"

ng = LoadNoduleGenerator(device=device, path=)

lung_img = Image.open("finalCXRDataset/controlimages/c0006.png")
centerx, centery = 640, 256
width = 40 
height = 50

lung_patch, mask = get_mask_image_patch(lung_img=lung_img, centerx=centerx, centery=centery, width=width, height=height)


nodule_imgs = ng.nodule_predict(masks=[mask], lung_patches=[lung_patch])

nodule_imgs[0].save("ayo.jpg")