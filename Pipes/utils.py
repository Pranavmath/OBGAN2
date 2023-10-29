from PIL import Image
import random
from scipy import stats
import numpy as np

# The Minimum value that a sum of a row/column has to be to be counted as part of the nodule and not part of the background
MIN_SUM = 7

centerposes = np.load("OBGAN2/centerposeshist.npy")
centerx_distrubution = stats.rv_histogram(np.histogram(centerposes[0], bins=500))
centery_distrubution = stats.rv_histogram(np.histogram(centerposes[1], bins=500))

# RGBA to RGB
def rgba_to_rgb(image):
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask = image.split()[3])
    return background

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

        background_image.paste(nodule, (centerx - width//2, centery - height//2), nodule)
  
    background_image.save("quandale.png")
    
    return rgba_to_rgb(background_image)


# Gets a random center x(s) and center y(s) to put the nodule(s) in (use the histogram of nodules in real images)
def get_centerx_getcentery(num_nodules):
    centerxs = centerx_distrubution.rvs(size=num_nodules)
    centerys = centery_distrubution.rvs(size=num_nodules)

    return [(int(centerxs[i].item()), int(centerys[i].item())) for i in range(len(centerxs))]


# Takes in PIL Image of nodule (+ its max size) and gets its dimension (width and height)
# Not including the black background around the nodule
def get_dim(nodule, max_size):
    nodule = np.array(nodule.convert("L"))
    ones = np.ones((1, max_size))

    # Sums of each column - 1, max_size
    sum_colums = ones @ nodule

    # Sum of each row - max_size, 1
    sum_rows = nodule @ (ones.T)

    width, height = 0, 0

    for i in range(max_size):
        if (sum_colums[0, i] > MIN_SUM):
            width += 1
        if (sum_rows[i, 0] > MIN_SUM):
            height += 1
    
    return (width, height)


# Takes in the current difficulty and the number of difficulties you want 
# Outputs those number of random difficulties at and above the current difficulty
# Uses a probability distrubution to give more weightage to the current difficulty 
# Decreases weightage to the right, no weightage to the left of curr diff because we haven't gotten to that point yet in curriculum learning).
def get_fake_difficulties(curr_difficulty, num_of_difficulties):
    if curr_difficulty == 1:
      curr_difficulty = 0.99
    return stats.truncweibull_min.rvs(a=curr_difficulty, size=num_of_difficulties, b=1, c=1)
