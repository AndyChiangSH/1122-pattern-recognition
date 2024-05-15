# HW3

Implement **ensemble methods** with Numpy and PyTorch.

# Usage

1. Move to this folder
    
    ```bash
    cd Homework/HW3/code
    ```
    
2. Train and test model
    
    ```bash
    python main.py
    ```
    
3. Lint the code
    
    ```bash
    flake8 main.py
    ```
    

# AdaBoost

1. Define the AdaBoost Classifier
    
    ```python
    clf_adaboost = AdaBoostClassifier(input_dim=X_train.shape[1])
    _ = clf_adaboost.fit(X_train, y_train, num_epochs=1000, learning_rate=0.001)
    ```
    
2. Plot the AUC curves of AdaBoost
   
   ![AUC curves of AdaBoost](code/AUC_curves/AdaBoost.png)

3. Plot the feature importance of AdaBoost
   
   ![feature importance of AdaBoost](code/feature_importance/AdaBoost.png)

# Bagging

1. Define the Bagging Classifier
    
    ```python
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=1000, learning_rate=0.004)
    ```
    
2. Plot the AUC curves of Bagging
      
   ![AUC curves of Bagging](code/AUC_curves/Bagging.png)

3. Plot the feature importance of Bagging
      
   ![feature importance of Bagging](code/feature_importance/Bagging.png)


# Decision Tree

1. Define the Decision Tree
    
    ```python
    clf_bagging = BaggingClassifier(input_dim=X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train, num_epochs=1000, learning_rate=0.004)
    ```
    
2. Plot the feature importance of Decision Tree
         
   ![feature importance of Decision Tree](code/feature_importance/DecisionTree.png)
