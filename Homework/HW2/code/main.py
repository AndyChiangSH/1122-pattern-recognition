import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0

        # Gradient Descent with Cross-Entropy Loss
        for epoch in range(self.num_iterations):
            # Linear Transformation
            linear_model = np.dot(inputs, self.weights) + self.intercept
            # Non-linear Transformation
            predictions = self.sigmoid(linear_model)

            # Compute the gradient of the loss
            loss = predictions - targets
            dw = (1 / n_samples) * np.dot(inputs.T, loss)
            db = (1 / n_samples) * np.sum(loss)

            # Update weights and intercept
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        # Linear Transformation
        linear_model = np.dot(inputs, self.weights) + self.intercept
        # Non-linear Transformation
        probabilities = self.sigmoid(linear_model)
        # Decision Making
        predictions = (probabilities >= 0.5).astype(int)
        
        return probabilities, predictions

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # Calculate class means
        self.m0 = np.mean(inputs[targets == 0], axis=0)
        self.m1 = np.mean(inputs[targets == 1], axis=0)

        # Compute within-class scatter matrix (SW)
        sw0 = np.cov(inputs[targets == 0], rowvar=False)
        sw1 = np.cov(inputs[targets == 1], rowvar=False)
        self.sw = sw0 + sw1

        # Compute between-class scatter matrix (SB)
        diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = np.dot(diff, diff.T)

        # Compute Fisher’s linear discriminant (w)
        self.w = np.linalg.inv(self.sw).dot(diff)

        # Calculate slope for plotting
        self.slope = -self.w[0] / self.w[1]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        # Project data onto 1D space
        projected = np.dot(inputs, self.w)

        # Compute distance to class means
        dist_to_m0 = np.abs(projected - np.dot(self.m0, self.w))
        dist_to_m1 = np.abs(projected - np.dot(self.m1, self.w))

        # Predict class based on closest mean
        predictions = dist_to_m0 > dist_to_m1

        return predictions.astype(int).flatten()

    def plot_projection(self, inputs: npt.NDArray[float]):
        # Plot data and projection line
        plt.scatter(inputs[:, 0], inputs[:, 1], c='gray', label='Data')
        plt.plot([-5, 5], [-5 * self.slope, 5 * self.slope], c='red')
        plt.title('Projection Line')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    # Sort the predictions by predicted probabilities
    sorted_indices = np.argsort(y_preds)
    y_trues_sorted = y_trues[sorted_indices]

    # Count the number of positive examples
    n_positive = np.sum(y_trues == 1)

    # Initialize the variables for AUC calculation
    auc = 0.0
    n_false_positives = 0
    n_true_positives = 0

    # Iterate through the sorted predictions
    for i in range(len(y_preds)):
        # Update the counts of false positives and true positives
        if y_trues_sorted[i] == 1:
            n_true_positives += 1
        else:
            n_false_positives += 1

        # Calculate the area under the ROC curve using the trapezoidal rule
        if i < len(y_preds) - 1 and y_preds[sorted_indices[i]] != y_preds[sorted_indices[i+1]]:
            auc += (n_true_positives / n_positive) * \
                (n_false_positives / (len(y_preds) - n_positive))

    return auc


def accuracy_score(y_trues, y_preds) -> float:
    # Count the number of correct predictions
    correct_predictions = np.sum(y_trues == y_preds)

    # Compute the accuracy score
    accuracy = correct_predictions / len(y_trues)

    return accuracy


def main():
    # Read data
    print("> Read data...")
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    # Part1: Logistic Regression
    print("> Part1: Logistic Regression...")
    LR = LogisticRegression(
        learning_rate=1e-2,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    print("> Part2: FLD...")
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    print("========== START ==========")
    main()
    print("=========== END ===========")
