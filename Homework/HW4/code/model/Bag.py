import torch
import torch.nn as nn


class BagModel(nn.Module):
    def __init__(self, instance_model):
        super(BagModel, self).__init__()
        self.instance_model = instance_model

    def forward(self, x):
        batch_size, num_instances, channels, height, width = x.size()
        # Flatten the instances into the batch dimension
        x = x.view(-1, channels, height, width)
        x = self.instance_model(x)
        # Reshape back to [batch_size, num_instances, num_classes]
        x = x.view(batch_size, num_instances, -1)
        x = torch.mean(x, dim=1)  # Aggregate the instance scores (mean)
        return x
