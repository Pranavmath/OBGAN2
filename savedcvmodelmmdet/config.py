import os

# The new config inherits a base config to highlight the necessary modification
_base_ = "/savedcvmodelmmdet/dino-4scale_r50_8xb2-12e_coco.py"
load_from = "/savedcvmodelmmdet/dino-4scale_r50_8xb2-12e_coco.pth"

# We also need to change the num_classes in head to match the dataset's annotation
# KEY: Needs to be changed based on each model
model = dict(bbox_head=dict(num_classes=1))

work_dir = "/content/modelcheckpoints"

# Modify dataset related settings
#data_root = '/content/ChestXRDataset/'
metainfo = {
  'classes': ('Nodule', )
}

param_scheduler =  param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.007, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[7, 11],
        gamma=0.1)
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='/content/ChestXRDataset/',
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root='/content/ChestXRDataset/',
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file='/content/ChestXRDataset/' + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
