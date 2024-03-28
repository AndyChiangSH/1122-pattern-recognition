import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # (X^T.X)^-1.X^T.Y
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000, batch_size: int = 1, alpha: float = 0.0):
        self.weights = np.random.rand(X.shape[1])
        self.intercept = 0
        losses = []
        n_samples = X.shape[0]
        L1_ratio = alpha * learning_rate

        # Training
        for _ in range(epochs):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Predict
                y_predicted = np.dot(X_batch, self.weights) + self.intercept
                y_error = y_batch - y_predicted

                # Compute gradients
                dw = -(1 / batch_size) * np.dot(X_batch.T, y_error)
                db = -(1 / batch_size) * np.sum(y_error)

                # L1 regularization term
                L1_term = L1_ratio * np.sign(self.weights)

                # Update parameters
                self.weights = self.weights - learning_rate * (dw + L1_term)
                self.intercept = self.intercept - learning_rate * db

            # Compute loss
            y_predicted = np.dot(X, self.weights) + self.intercept
            loss = compute_mse(y_predicted, y)
            losses.append(loss)

        return losses

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)


def main():
    # Load training data
    train_df = pd.read_csv('./data/train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    # scikit-learn is only for checking
    LR_SL = LinearRegression()
    LR_SL.fit(train_x, train_y)
    logger.info(f'{LR_SL.coef_=}, {LR_SL.intercept_=:.4f}')

    # Train Linear Regression Model - Closed-form Solution
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    # Train Linear Regression Model - Gradient Descent Solution
    LR_GD = LinearRegressionGradientdescent()
    # Without L1 regularization
    # losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=30000, batch_size=64)
    # With L1 regularization
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=30000, batch_size=64, alpha=0.01)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    # Training result
    logger.info('Train:')
    y_preds_cf = LR_CF.predict(train_x)
    y_preds_gd = LR_GD.predict(train_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, train_y)
    mse_gd = compute_mse(y_preds_gd, train_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')

    # Load testing data
    test_df = pd.read_csv('./data/test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    # Testing result
    logger.info('Test:')
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
