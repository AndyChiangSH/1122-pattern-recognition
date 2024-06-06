import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split

from dataset import BagDataset


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add arguments
    parser.add_argument('--dataset_path', type=str, default='./dataset', help="Path to dataset")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch size")

    # Parse the arguments
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    print("> Start training!")
    
    # Parse the arguments
    args = parse_args()
    
    # Load the training dataset
    print("> Load the training dataset...")
    classes = [0, 1]
    bags = []
    labels = []
    
    # For each class
    for class_ in classes:
        train_dataset_path = os.path.join(args.dataset_path, "train", f"class_{class_}")
        train_dataset_files = os.listdir(train_dataset_path)
        print(f">> Load the {train_dataset_path}...")
        
        # For each file
        for train_dataset_file in train_dataset_files:
            train_dataset_file_path = os.path.join(train_dataset_path, train_dataset_file)
            
            # Load the file
            with open(train_dataset_file_path, 'rb') as f:
                data = pickle.load(f)
                
            bags.append(data)
            labels.append(class_)
            
    # print("len(bags):", len(bags))
    # print("len(labels):", len(labels))
        
    # Split the dataset into training and validation sets
    train_bags, valid_bags, train_labels, valid_labels = train_test_split(
        bags, labels, test_size=0.1
    )
    
    # Create BagDataset
    train_dataset = BagDataset(train_bags, train_labels)
    valid_dataset = BagDataset(valid_bags, valid_labels)
    
    # print("len(train_dataset):", len(train_dataset))
    # print("len(valid_dataset):", len(valid_dataset))
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    # print("len(train_loader):", len(train_loader))
    # print("len(valid_loader):", len(valid_loader))

    for bags, labels in train_loader:
        # print("bags.shape:", bags.shape)
        # print("labels.shape:", labels.shape)
