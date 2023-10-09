from data import OBData, collate_fn, NoduleData
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from discriminator import Discriminator
from torch import optim
from torch import nn
from time import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np 

dataset = NoduleData("finalCXRDataset/final.csv", "finalCXRDataset/images", 140)

dataset[100][0].show()

"""
writer = SummaryWriter()

dataset = OBData("finalCXRDataset/final.csv", "finalCXRDataset/images")

data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

difficulty = []

for i, data in enumerate(data_loader):
    for i in data[2]:
        difficulty.append(i.item())

plt.hist(difficulty, 100, (0, 5), True)
plt.xticks(np.linspace(0, 5, 11))
plt.show()


disc = Discriminator()
optimizer = optim.Adam(disc.parameters())
criterion = nn.MSELoss()
num_epochs = 2

prog_bar = tqdm(data_loader, total=len(data_loader))

for epoch in range(num_epochs):
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        # Dummy expected data with size: (batch_size, 1)
        y_expected = torch.ones(len(data[0]), 1) 

        images = data[0]
        images = torch.stack([image for image in images])

        y_pred = disc(images)
        loss = criterion(y_pred, y_expected)

        writer.add_scalar("Loss/train", loss, epoch)

        loss.backward()
        optimizer.step()

writer.flush()
writer.close()
"""