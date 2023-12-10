from data import OBData

# The maximum size of a nodule (length)
MAX_SIZE = 140
# Starting Difficulty
START_DIFF = 0.4275
# Step in difficulty when training the model (change in difficulty must be negative beacause smaller difficulty means it's harder)
STEP = 0.0167
# Ending Difficulty
END_DIFF = 0.01
# Number of Fake Images at the starting difficulty
START_NUM_FAKE = 400
# Number of Fake Images at the ending difficulty
END_NUM_FAKE = 500
# Number of epochs for each step in difficulty
NUM_EPOCHS_FOR_STEP = 2
# Batch Size
BATCH_SIZE = 10


# ------------------------------------- ACTUAL CODE ---------------------------------------------

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

# Current (Start) Difficulty
curr_diff = START_DIFF

print("Starting")

sum_images_trained = 0

while curr_diff >= END_DIFF:
    real_images_bboxes = len(ob_dataset.all_above_difficulty(curr_diff))
    control_images_bboxes = len(ob_dataset.get_control_images(num=len(real_images_bboxes)))
    all_images_bboxes = real_images_bboxes + control_images_bboxes
    
    if curr_diff <= 0.07:
        num_epochs = 4
    else:
        num_epochs = NUM_EPOCHS_FOR_STEP
    
    for e in range(num_epochs):
        sum_images_trained += all_images_bboxes
    
    print(sum_images_trained)

    # Steps the current difficulty down (makes it harder)
    curr_diff -= STEP