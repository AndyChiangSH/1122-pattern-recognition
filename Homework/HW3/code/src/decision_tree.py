import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None


    def fit(self, X, y):
        """Builds the decision tree based on the training data."""
        self.tree = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
        """Recursively grows a binary decision tree."""
        # num_samples, num_features = X.shape
        # Base cases: if dataset is pure, or depth limit has been reached
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            most_common_label = np.bincount(y).argmax()
            return {'label': most_common_label}

        # Find the best split
        best_feature, best_threshold = find_best_split(X, y)

        # Grow the children recursively
        left_indices, right_indices = split_dataset(
            X, best_feature, best_threshold)
        left_child = self._grow_tree(
            X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(
            X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': best_feature, 'threshold': best_threshold, 'left': left_child, 'right': right_child}


    def predict(self, X):
        """Predicts the class labels for the given samples."""
        # Predict each sample with decision tree
        return np.array([self._predict_tree(x, self.tree) for x in X])


    def _predict_tree(self, x, tree_node):
        """Predicts a single sample using the decision tree."""
        # Base case: if the node is a leaf node, return its label
        if 'label' in tree_node:
            return tree_node['label']
        
        # Recursively traverse the decision tree based on the feature and threshold
        feature_index = tree_node['feature_index']
        threshold = tree_node['threshold']
        
        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])


def split_dataset(X, feature_index, threshold):
    """Splits the dataset into left and right subsets based on the threshold of the feature."""
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    
    return left_indices, right_indices


def find_best_split(X, y):
    """Finds the best split by iterating over every feature and every threshold."""
    best_entropy = float('inf')
    best_feature, best_threshold = None, None
    num_samples, num_features = X.shape

    # For each feature
    for feature_index in range(num_features):
        thresholds = np.unique(X[:, feature_index])
        
        # For each threshold of the feature
        for threshold in thresholds:
            # Split the dataset into left and right subsets based on the current feature and threshold
            left_indices, right_indices = split_dataset(X, feature_index, threshold)
            # Calculate the weighted average of entropy
            entropy_value = calculate_weighted_entropy(y[left_indices], y[right_indices])
            # Minimize the entropy
            if entropy_value < best_entropy:
                best_entropy = entropy_value
                best_feature, best_threshold = feature_index, threshold

    return best_feature, best_threshold


def entropy(y):
    """Calculates the entropy of an array of class labels."""
    # Calculate the probabilities of each class label
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Return the entropy of the class labels
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_weighted_entropy(left_y, right_y):
    """Calculates the weighted average of entropy for two splits."""
    # Calculate the probabilities of right and left subsets
    total_count = len(left_y) + len(right_y)
    p_left = len(left_y) / total_count
    p_right = len(right_y) / total_count
    
    # Return the weighted average of entropy
    return p_left * entropy(left_y) + p_right * entropy(right_y)
