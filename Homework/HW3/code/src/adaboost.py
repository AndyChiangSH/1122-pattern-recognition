import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10):
        # Initialize AdaBoost with a list of weak learners
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(num_learners)]
        self.sample_weights = None
        self.alphas = []

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int = 500, learning_rate: float = 0.001):
        # Determine the number of samples
        n_samples = X_train.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples
        losses_of_models = []

        # Loop over each weak learner
        for learner in self.learners:
            # Initialize the optimizer with learning rate for gradient descent
            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)
            # Training loop for the specified number of epochs
            for epoch in range(num_epochs):
                learner.train()
                optimizer.zero_grad()

                outputs = learner(X_train).squeeze()
                loss = entropy_loss(outputs, y_train)
                weighted_loss = torch.mean(torch.tensor(self.sample_weights, dtype=torch.float32) * loss)

                weighted_loss.backward()
                optimizer.step()

            with torch.no_grad():
                learner.eval()
                # Convert predictions to 0 or 1 using threshold
                predictions = (learner(X_train).squeeze() >= 0.5).float()
                incorrect = (predictions != y_train).float()
                weighted_error = np.dot(self.sample_weights, incorrect.numpy())

                # Handle division by zero in alpha calculation
                if weighted_error == 0 or weighted_error == 1:
                    alpha = 1e-10  # Small alpha to avoid infinite weights
                else:
                    alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)

                self.alphas.append(alpha)

                # Update sample weights exponentially
                exp_component = np.exp(-alpha * (2 * y_train.numpy() - 1) * (2 * predictions.numpy() - 1))
                self.sample_weights *= exp_component
                self.sample_weights /= np.sum(self.sample_weights)

                # Record the loss of this epoch
                losses_of_models.append(weighted_loss.item())

        return losses_of_models

    def predict_learners(self, X):
        final_output = np.zeros(X.shape[0])
        predictions = []
        # Aggregate predictions from all learners weighted by alpha
        for learner, alpha in zip(self.learners, self.alphas):
            prediction = learner(X).detach().squeeze()
            predictions.append(prediction.numpy())
            final_output += alpha * prediction.numpy()

        return final_output, predictions

    def compute_feature_importance(self):
        importance = np.zeros(self.learners[0].linear.weight.shape[1])
        # Calculate the importance of each feature across all learners
        for learner, alpha in zip(self.learners, self.alphas):
            importance += np.abs(learner.linear.weight.detach().numpy().squeeze() * alpha)

        return importance / len(self.learners)
