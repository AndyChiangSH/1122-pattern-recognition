import torch
import torch.nn as nn


class BagModel_1(nn.Module):
    def __init__(self, instance_model):
        super(BagModel_1, self).__init__()
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
        x = torch.mean(x, dim=1)
        # x = torch.max(x, dim=1).values
        # print("x.shape:", x.shape)
        
        return x


class BagModel_2(nn.Module):
    def __init__(self, instance_model, num_features):
        super(BagModel_2, self).__init__()
        self.instance_model = instance_model
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        batch_size, num_instances, channels, height, width = x.size()

        # Flatten the instances into the batch dimension
        x = x.view(-1, channels, height, width)

        # Extract features for each instance
        features = self.instance_model(x)

        # Reshape back to [batch_size, num_instances, num_features]
        features = features.view(batch_size, num_instances, -1)
        # print("features.shape:", features.shape)

        # Aggregate the instance features (max-pooling)
        instance_scores, _ = torch.max(features, dim=1)
        # print("instance_scores.shape:", instance_scores.shape)

        # Classify the bag based on the aggregated features
        bag_scores = self.fc(instance_scores)

        return bag_scores
