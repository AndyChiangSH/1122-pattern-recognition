import torch
import torch.nn as nn


class BagModel(nn.Module):
    def __init__(self, instance_model):
        super(BagModel, self).__init__()
        self.instance_model = instance_model

    def forward(self, x):
        # print("x.shape:", x.shape)
        batch_size, num_instances, channels, height, width = x.size()
        # Flatten the instances into the batch dimension
        x = x.view(-1, channels, height, width)
        # Input the model
        x = self.instance_model(x)
        # Reshape back to [batch_size, num_instances, num_classes]
        x = x.view(batch_size, num_instances, -1)
        # Aggregate the instance scores (mean)
        # print("x.shape:", x.shape)
        # x = torch.mean(x, dim=1)
        x = torch.max(x, dim=1).values
        # print("x.shape:", x.shape)
        
        return x
