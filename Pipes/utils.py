from PIL import Image, ImageDraw
import random
from scipy import stats
import numpy as np

mask = Image.open("OBGAN2/mask.png").convert("L").resize((140, 140))

# The Minimum value that a sum of a row/column has to be to be counted as part of the nodule and not part of the background
MIN_SUM = 70

# The Padding value used to prepare the dataset for training the nodule generator
PAD = 30

centerposes = np.load("OBGAN2/centerposeshist.npy")
centerx_distrubution = stats.rv_histogram(np.histogram(centerposes[0], bins=500))
centery_distrubution = stats.rv_histogram(np.histogram(centerposes[1], bins=500))

# ------------------------------------------------------------------------------------

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batch_data(l, b_size):
    new_l = []

    batched_images = list(batch([ele[0] for ele in l], b_size))
    batched_bboxes = list(batch([ele[1] for ele in l], b_size))

    for i in range(len(batched_images)):
        new_l.append((batched_images[i], batched_bboxes[i]))
    
    return new_l


# Place the center of each nodule image at a certain location on the background image (transperantly)
# Nodules and Background image are PIL Images
def place_nodules(background_image, nodules, center_xy_nodules):
    assert len(nodules) == len(center_xy_nodules)

    background_image = background_image.convert("L")

    for nodule_idx in range(len(nodules)):
        centerx, centery = center_xy_nodules[nodule_idx]
        nodule = nodules[nodule_idx]
        width, height = nodule.size

        nodule = nodule.convert("L")

        background_image.paste(nodule, (centerx - width//2, centery - height//2), mask)
      
    return background_image.convert("RGB")


# Gets a random center x(s) and center y(s) to put the nodule(s) in (use the histogram of nodules in real images)
def get_centerx_getcentery(num_nodules):
    centerxs = centerx_distrubution.rvs(size=num_nodules)
    centerys = centery_distrubution.rvs(size=num_nodules)

    return [(int(centerxs[i].item()), int(centerys[i].item())) for i in range(len(centerxs))]


# Takes in PIL Image of nodule (+ its max size) and gets its dimension (width and height)
# Not including the black background around the nodule
def get_dim(nodule, max_size):
    nodule = np.array(nodule.convert("L")) / 255
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


def get_width_and_height(diff):
    nodule_area =  diff * (140 ** 2)
    square_length = nodule_area ** 0.5

    width = random.randint(int(square_length / 1.4), int(square_length))
    height = nodule_area // width

    return width, height


# Gets the mask and the image patch (from the control lung image) without nodules
# Goes into nodule inpainter to add nodule
def get_mask_image_patch(lung_img, centerx, centery, width, height):
    if height >= width:
        l = 2 * PAD + height
        location_nodule = [(l - width) // 2, PAD, (l - width) // 2 + width, height + PAD]
        
    if width > height:
        l = 2 * PAD + width
        location_nodule = [PAD, (l - height) // 2, width + PAD, (l - height) // 2 + height]
    
    nodule_img = lung_img.crop([centerx - l // 2, centery - l // 2, centerx + l // 2, centery + l // 2])
    
    mask = Image.new(mode = "RGB", size = (l, l), color = (255, 255, 255))
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(location_nodule, fill = "#000000")

    return nodule_img, mask