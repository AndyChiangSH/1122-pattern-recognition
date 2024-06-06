import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import BagDataset
from model.CNN import SimpleCNN


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training arguments")

    # Add arguments
    parser.add_argument('--dataset_path', type=str, default='./dataset', help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")

    # Parse the arguments
    args = parser.parse_args()
    
    return args


def load_data(dataset_path, classes):
    bags = []
    labels = []

    # For each class
    for class_ in classes:
        train_dataset_path = os.path.join(
            dataset_path, "train", f"class_{class_}")
        train_dataset_files = os.listdir(train_dataset_path)
        print(f">> Load the {train_dataset_path}...")

        # For each file
        for train_dataset_file in train_dataset_files:
            train_dataset_file_path = os.path.join(
                train_dataset_path, train_dataset_file)

            # Load the file
            with open(train_dataset_file_path, 'rb') as f:
                data = pickle.load(f)

            bags.append(data)
            labels.append(class_)

    return bags, labels


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Train", position=1):
        # print("inputs.shape:", inputs.shape)
        # print("labels.shape:", labels.shape)
        # Adjust dimensions to [batch_size, channels, depth, height, width]
        # inputs = inputs.permute(0, 2, 1, 3, 4)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    
    return train_loss


def valid_model(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Valid", position=1):
            # Adjust dimensions to [batch_size, channels, depth, height, width]
            # inputs = inputs.permute(0, 2, 1, 3, 4)
            outputs = model(inputs)
            # print("outputs:", outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            # print("preds:", preds)
            # print("labels:", labels)
            correct_predictions += torch.sum(preds == labels)

    valid_loss = running_loss / len(valid_loader.dataset)
    valid_accuracy = correct_predictions.double() / len(valid_loader.dataset)
    
    return valid_loss, valid_accuracy


if __name__ == '__main__':
    print("> Start training!")
    
    # Parse the arguments
    args = parse_args()
    
    # Load the training dataset
    print("> Load the training dataset...")
    classes = [0, 1]
    bags, labels = load_data(args.dataset_path, classes)
            
    # print("len(bags):", len(bags))
    # print("len(labels):", len(labels))
        
    # Split the dataset into training and validation sets
    train_bags, valid_bags, train_labels, valid_labels = train_test_split(
        bags, labels, test_size=0.1
    )
    
    # Create BagDataset
    print("> Create BagDataset...")
    train_dataset = BagDataset(train_bags, train_labels)
    valid_dataset = BagDataset(valid_bags, valid_labels)
    
    # print("len(train_dataset):", len(train_dataset))
    # print("len(valid_dataset):", len(valid_dataset))
    
    # Create DataLoader
    print("> Create DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    print("len(train_loader):", len(train_loader))
    print("len(valid_loader):", len(valid_loader))

    # Initialize the model, loss function, and optimizer
    print("> Initialize the model, loss function, and optimizer...")
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.epochs), desc="Epoch", position=0):
        # Train the model
        # print(">> Train the model...")
        train_loss = train_model(model, train_loader, criterion, optimizer)

        # Valid the model
        # print(">> Valid the model...")
        valid_loss, valid_accuracy = valid_model(model, valid_loader, criterion)
        
        print(f'\nEpoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}')
    
    print("> Stop training!")
