import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from loguru import logger

from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, plot_feature_importance


def main():
    # Load training and testing data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Prepare features and targets
    # (n_samples, n_features)
    X_train = train_df.drop(['target'], axis=1).to_numpy()
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    X_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()
    
    feature_names = train_df.drop(['target'], axis=1).columns.to_list()

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1])
    _ = clf_adaboost.fit(X_train, y_train, num_epochs=500, learning_rate=0.001)
    y_pred_probs = clf_adaboost.predict_learners(X_test)
    # Assuming output needs to be binarized
    y_pred_classes = np.sign(y_pred_probs)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=[y_pred_probs], y_trues=y_test,fpath='./AUC_curves/AdaBoost.png')
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(feature_importance, feature_names, fpath='./feature_importance/AdaBoost.png')
    # logger.info(f'AdaBoost - Feature Importance: {feature_importance}')

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=500, learning_rate=0.001)
    y_pred_probs = clf_bagging.predict_learners(X_test)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)  # Assuming output is probability
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')

    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred_classes)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')


if __name__ == '__main__':
    main()
