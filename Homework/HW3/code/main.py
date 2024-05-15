import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from loguru import logger
import torch

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, plot_feature_importance
from src.decision_tree import gini_index, entropy


def main():
    # Load training and testing data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Prepare features and targets, converting to tensors
    X_train = train_df.drop(['target'], axis=1).values
    y_train = train_df['target'].values
    X_test = test_df.drop(['target'], axis=1).values
    y_test = test_df['target'].values
    
    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Get feature names
    feature_names = train_df.drop(['target'], axis=1).columns.to_list()

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1])
    _ = clf_adaboost.fit(X_train, y_train, num_epochs=1000, learning_rate=0.001)
    y_pred_probs, y_predictions = clf_adaboost.predict_learners(X_test)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_predictions, y_trues=y_test, fpath='./AUC_curves/AdaBoost.png')
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./feature_importance/AdaBoost.png')

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=1000, learning_rate=0.004)
    y_pred_probs, y_predictions = clf_bagging.predict_learners(X_test)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_predictions, y_trues=y_test, fpath='./AUC_curves/Bagging.png')
    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./feature_importance/Bagging.png')

    # Decision Tree
    test_array = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    logger.info(f'DecisionTree - Gini index: {gini_index(test_array):.4f}')
    logger.info(f'DecisionTree - Entropy: {entropy(test_array):.4f}')
    
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')
    feature_importance = clf_tree.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./feature_importance/DecisionTree.png')


if __name__ == '__main__':
    main()
