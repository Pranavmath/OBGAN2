from PIL import Image
import random

# Place the center of each nodule image at a certain location on the background image (transperantly)
# Nodules and Background image are PIL Images
def place_nodules(background_image, nodules, center_xy_nodules):
    assert len(nodules) == len(center_xy_nodules)

    background_image = background_image.convert("RGBA")

    for nodule_idx in range(len(nodules)):
        centerx, centery = center_xy_nodules[nodule_idx]
        nodule = nodules[nodule_idx]
        width, height = nodule.size

        nodule = nodule.convert("RGBA")

        background_image.paste(nodule, (centerx - width//2, centery - height//2))
    
    return background_image

# Gets a random center x(s) and center y(s) to put the nodule(s) in (use the histogram of nodules in real images)
def get_centerx_getcentery(num_nodules):
    pass 