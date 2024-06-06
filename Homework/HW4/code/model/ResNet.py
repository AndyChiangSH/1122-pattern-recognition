import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    # Change num_classes to 1 for binary classification with BCEWithLogitsLoss
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights=True)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
