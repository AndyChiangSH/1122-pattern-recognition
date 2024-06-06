import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv3d(256, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Adjust based on the output size of the conv layers
        self.fc1 = nn.Linear(196608, 128)
        # Assuming binary classification
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
