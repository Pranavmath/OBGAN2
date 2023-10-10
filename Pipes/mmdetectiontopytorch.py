# -*- coding: utf-8 -*-
"""MMdetectionToPytorch

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PujdC-vCAcixm7bplVmCkYWsgHFI6UgW
"""

import torch
import torchvision.transforms.functional as F
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

def predict_cv(model, img, gt_bboxes, gt_labels):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)

  torch_img = F.pil_to_tensor(img)
  x = torch.stack([torch_img.to(device).float()])

  """
  The DetDataSample must follow this format:
  <DetDataSample(

      META INFORMATION
      img_shape: _
      scale_factor: _

      DATA FIELDS
      batch_input_shape: _
      gt_instances: <InstanceData(

              META INFORMATION

              DATA FIELDS
              bboxes: _
              labels: _
          ) at _>
  ) at _>
  """
  y = DetDataSample(metainfo={"img_shape": img.size, "scale_factor": (1, 1)})
  y.batch_input_shape = img.size

  gt_instances = InstanceData()
  gt_instances.bboxes = torch.tensor(gt_bboxes).to(device)
  gt_instances.labels = torch.tensor(gt_labels).to(device)

  y.gt_instances = gt_instances

  loss = model.loss(x, [y])

  return loss
