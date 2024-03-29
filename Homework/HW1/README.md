# HW1

Implement **linear regression** by only using numpy.

1. Move to this folder.
    
    ```bash
    cd Homework/HW1/code
    ```
    
2. Train the Linear Regression Model with Closed-form Solution
    
    ```python
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    ```
    
3. Train the Linear Regression Model with Gradient Descent Solution
    
    ```python
    LR_GD = LinearRegressionGradientdescent()
    # Without L1 regularization
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=30000, batch_size=64)
    # With L1 regularization
    losses = LR_GD.fit(train_x, train_y, learning_rate=1e-4, epochs=30000, batch_size=64, alpha=0.01)
    LR_GD.plot_learning_curve(losses)
    ```
    
4. Train and test model
    
    ```bash
    python main.py
    ```
    
5. Lint the code
    
    ```bash
    flake8 main.py
    ```
    
6. Test the code
    
    ```bash
    pytest ./test_main.py -s
    ```