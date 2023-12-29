from currlearningmodels.nodulegen import LoadNoduleGenerator
from utils import get_mask_image_patch
from PIL import Image

device = "cuda"

ng = LoadNoduleGenerator(device=device, path="/content/50000.pth")

lung_img = Image.open("OBGAN2/finalCXRDataset/controlimages/c0007.png")
centerx, centery = 640, 256
width = 34
height = 53

lung_patch, mask = get_mask_image_patch(lung_img=lung_img, centerx=centerx, centery=centery, width=width, height=height)

lung_patch.save("withoutnodule.jpg")

nodule_imgs = ng.nodule_predict(masks=[mask], lung_patches=[lung_patch])

nodule_imgs[0].save("withnodule.jpg")