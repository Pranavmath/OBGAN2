import torch
from torch import nn

class DifficultyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 7)
        self.max1 = nn.MaxPool2d(6)
        self.conv2 = nn.Conv2d(10, 32, 5)
        self.max2 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(32, 100, 4)
        self.max3 = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fcn1 = nn.Linear(8100, 100)
        self.fcn2 = nn.Linear(100, 1)
        self.drop1 = nn.Dropout(0.6)
        self.drop2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)   
        x = self.max3(x)

        x = self.flat(x)

        x = self.drop1(x)
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fcn2(x)
        x = self.sigmoid(x)

        return x

"""
df = DifficultyPredictor()
x = torch.randn((1, 3, 1024, 1024))
y = df(x)
print(y)
"""