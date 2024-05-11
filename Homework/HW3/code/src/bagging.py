import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # Creating a list of 10 weak classifiers as specified.
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(10)]

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int = 100, learning_rate: float = 0.01):
        # Initialize an empty list to store the loss of each model after training
        losses_of_models = []
        # Number of samples in the training data
        n_samples = X_train.shape[0]

        # Loop over each learner in the ensemble
        for model in self.learners:
            # For each learner, draw a bootstrap sample (sampling with replacement) from the training data
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            # Set up the optimizer for the current learner
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop for the current learner
            for epoch in range(num_epochs):
                model.zero_grad()
                outputs = model(X_sample).squeeze()
                loss = entropy_loss(outputs, y_sample.float())
                loss.backward()
                optimizer.step()

            # Append the final loss of this model to the list of losses
            losses_of_models.append(loss.item())

        return losses_of_models


    def predict_learners(self, X: torch.Tensor) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        # Aggregate predictions from all learners
        model_outputs = []
        
        # Each model predicts the output, which are then aggregated to produce a final prediction
        for model in self.learners:
            model_outputs.append(model(X).detach().sigmoid().numpy())
        
        # Average predictions across all learners
        return np.mean(model_outputs, axis=0)

    def compute_feature_importance(self) -> t.Sequence[float]:
        # Compute the average feature importance across all learners
        importance = np.zeros(self.learners[0].linear.weight.shape[1])
        
        # Feature importance can be estimated from the weights in a linear model
        for model in self.learners:
            importance += np.abs(model.linear.weight.detach().numpy())
        
        # Normalize by the number of learners to get the average importance
        return importance / len(self.learners)
