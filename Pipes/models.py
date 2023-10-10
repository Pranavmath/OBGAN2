import torch
from torch import nn

class Discriminator(nn.Module):
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
        self.relu = nn.ReLU()
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

class NoduleGenerator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(NoduleGenerator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 8, 1, 0),
            self._block(features_g * 16, features_g * 8, 5, 2, 1),
            self._block(features_g * 8, features_g * 4, 5, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

        self.fcn = nn.Linear(100, 100)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, label):
      x = self.fcn(noise)
      x = torch.add(x, label)

      x = torch.unsqueeze(x, -1)
      x = torch.unsqueeze(x, -1)

      return self.net(x)
