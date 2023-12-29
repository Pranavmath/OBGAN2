from data import OBData
from PIL import Image, ImageDraw
from tqdm import tqdm

ob_dataset = OBData(csv="finalCXRDataset/final.csv", img_dir="finalCXRDataset/images", control_img_dir="finalCXRDataset/controlimages")

data = ob_dataset.all_below_difficulty(0.51)

i = 0

PAD = 30

for image, nodules in tqdm(data):
    for nodule in nodules:
        xmin, ymin, xmax, ymax = nodule
        width, height = xmax-xmin, ymax-ymin

        if height >= width:
            l = 2 * PAD + height
            crop = [xmin - (l - width) // 2, ymin - PAD, xmax + (l - width) // 2, ymax + PAD]
            location_nodule = [(l - width) // 2, PAD, (l - width) // 2 + width, height + PAD]
        
        if width > height:
            l = 2 * PAD + width
            crop = [xmin - PAD, ymin - (l - height) // 2, xmax + PAD, ymax + (l - height) // 2]
            location_nodule = [PAD, (l - height) // 2, width + PAD, (l - height) // 2 + height]


        nodule_img = image.crop(crop)

        mask = Image.new(mode = "RGB", size = (l, l), color = (255, 255, 255))
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(location_nodule, fill = "#000000")


        nodule_img = nodule_img.resize((256, 256))
        mask = mask.resize((256, 256))


        nodule_img.save(f"nodulegendataset/nodules/{i}.jpg")
        mask.save(f"nodulegendataset/masks/{i}.jpg")
        i += 1

        nodule_img = nodule_img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        nodule_img.save(f"nodulegendataset/nodules/{i}.jpg")
        mask.save(f"nodulegendataset/masks/{i}.jpg")
        i += 1
