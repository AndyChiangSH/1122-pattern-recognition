import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        # Create 10 weak learners
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(num_learners)]
        # All samples initially have equal weights
        self.sample_weights = None
        # Initialize alpha values
        self.alphas = []


    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int = 500, learning_rate: float = 0.001):
        # Initialize
        n_samples = X_train.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples
        losses_of_models = []

        # Train for each weak learner
        for learner in self.learners:
            optimizer = optim.Adam(learner.parameters(), lr=learning_rate)
            
            # Train for each epoch
            for epoch in range(num_epochs):
                outputs = learner(X_train).squeeze()
                loss = entropy_loss(outputs, y_train)
                weighted_loss = torch.mean(torch.tensor(self.sample_weights) * loss)
                
                weighted_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Calculate errors and update weights
            with torch.no_grad():
                # Predict on training set
                learner.eval()
                predictions = torch.sign(learner(X_train).squeeze())
                incorrect = (predictions != y_train).float()
                weighted_error = np.dot(self.sample_weights, incorrect.numpy())

                # Avoid division by zero
                # if weighted_error == 0:
                #     continue
                
                # Alpha calculation
                # alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
                
                print("weighted_error:", weighted_error)

                if weighted_error <= 0 or weighted_error >= 1:
                    # Large positive or negative value depending on sign
                    alpha = np.sign(0.5 - weighted_error)
                else:
                    alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
                
                self.alphas.append(alpha)

                # Update weights
                self.sample_weights *= np.exp(-alpha * y_train.numpy() * predictions.numpy())
                # Normalize weights
                self.sample_weights /= np.sum(self.sample_weights)

                # Store model's loss
                losses_of_models.append(weighted_loss.item())

        return losses_of_models


    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        # Aggregate predictions from all learners
        predictions = []
        final_output = np.zeros(X.shape[0])
        
        # Test for each weak learner
        for learner, alpha in zip(self.learners, self.alphas):
            prediction = learner(X).detach().squeeze()
            predictions.append(prediction.numpy())
            final_output += alpha * prediction.numpy()
            
        return np.sign(final_output), predictions


    def compute_feature_importance(self) -> t.Sequence[float]:
        # This assumes the feature importance is the sum of the absolute weights of the model parameters
        importance = np.zeros(self.learners[0].linear.weight.shape[1])
        
        # Calculate feature importance for each learner
        for learner, alpha in zip(self.learners, self.alphas):
            importance += np.abs(learner.linear.weight.detach().numpy().squeeze() * alpha)
            
        print("alpha:", self.alphas)
            
        # Average feature importance over all learners
        return importance / len(self.learners)
