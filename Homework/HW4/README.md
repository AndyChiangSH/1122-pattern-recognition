# HW4

Convolutional Neural Networks for Bag Classification

## Setup environment

1. Move to this folder (optional)
    
    ```bash
    cd Homework/HW4/
    ```
    
2. Create a virtual environment
    
    ```bash
    conda env create -f environment.yml
    ```
    
3. Activate the virtual environment
    
    ```bash
    conda activate 1122-pattern-recognition
    ```
    

## Train

1. Put the dataset in the `dataset/` folder
2. Train the model
    
    ```bash
    python ./code/train.py --name="test" --model="ResNet18" --epochs=50 --learning_rate=0.000005 --batch_size=4
    ```
    
    - `--name`: model name
    - `--model`: model type
    - `--epochs`: number of training epochs
    - `--batch_size`: number of batch size
    - `--learning_rate`: learning rate
3. The log will be saved in the `log/` folder
4. The model will be saved in the `model/` folder

## Test

1. You can download the model from [here](https://drive.google.com/file/d/1cgLc9vYbvntiK5ZsJmMVQSeT-GsS-NhE/view?usp=sharing) (optional)
2. Put the model in the `model/` folder
3. Test the model
    
    ```bash
    python ./code/test.py --name="test"
    ```
    
    - `--name`: model name
4. The submission file (.csv) will be saved in the `submission/` folder
5. Upload the submission file (.csv) to the [Kaggle competition](https://www.kaggle.com/competitions/nycu-ml-pattern-recognition-hw-4/overview)