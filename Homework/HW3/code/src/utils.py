import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
    """
    Use PyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation allowed.
    """

    def __init__(self, input_dim, num_layers=1):
        super(WeakClassifier, self).__init__()
        self.num_layers = num_layers
        if self.num_layers == 1:
            # One layer, outputs directly the prediction
            self.linear = nn.Linear(input_dim, 1)
        elif self.num_layers == 2:
            # Two layers, first transforms the input
            self.linear1 = nn.Linear(input_dim, input_dim)
            # Second layer outputs the prediction
            self.linear2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        elif self.num_layers == 2:
            x = self.linear1(x)
            return self.linear2(x)


def entropy_loss(outputs, targets):
    """
    Computes the binary cross-entropy loss between the target and output logits.
    """
    # Calculate the cross entropy loss
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)


def plot_learners_roc(y_preds: t.List[t.Sequence[float]], y_trues: t.Sequence[int], fpath='./AUC_curves/tmp.png'):
    """
    Plots the AUC curves of each learner and saves the plot to a file.
    """
    plt.figure()
    
    # Plot the ROC curve of each learner
    for y_pred in y_preds:
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f})')
    
    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('AUC curves of each weak classifier')
    plt.legend(loc="lower right")
    
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(importances, feature_names, fpath='./feature_importance/tmp.png'):
    # Create arrays from the list to enable easy sorting
    importances, feature_names = zip(*sorted(zip(importances, feature_names)))

    # Creating the bar plot
    plt.figure()
    plt.barh(feature_names, importances, color='blue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    
    plt.savefig(fpath)
    plt.close()
