from data import OBData
from PIL import Image, ImageDraw 


ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

print("Starting")

sum = 0

for i in range(len(ob_dataset)):
    img, nodules = ob_dataset[i]

    if len(nodules) >= 3:
        if sum >= 1:
            for nodule in nodules:
                img1 = ImageDraw.Draw(img)
                img1.rectangle(nodule, outline="red", width=4)

            img.show()

            break

        sum += 1