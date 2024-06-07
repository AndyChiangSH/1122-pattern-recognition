import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pickle
import argparse
import os
from tqdm import tqdm

from dataset import BagDataset
from model.Bag import BagModel
from model.CNN import CNN
from model.ResNet import ResNet


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training arguments")

    # Add arguments
    parser.add_argument('--name', type=str, default="test", help="Model name")
    parser.add_argument('--model', type=str, default="test", help="Model type")
    parser.add_argument('--dataset_path', type=str, default='./dataset', help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization weight decay")

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
    correct_predictions = 0
    total_samples = 0
    for inputs, labels in tqdm(train_loader, desc="Train", position=1):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        preds = torch.round(torch.sigmoid(outputs))
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        
        # print("outputs:", outputs)
        # print("preds:", preds)
        # print("labels:", labels)
        # print("loss.item():", loss.item())

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct_predictions / total_samples
    
    return train_loss, train_accuracy


def valid_model(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Valid", position=1):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            # print("outputs:", outputs)
            # print("preds:", preds)
            # print("labels:", labels)
            # print("loss.item():", loss.item())

    valid_loss = running_loss / len(valid_loader.dataset)
    valid_accuracy = correct_predictions / total_samples
    
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
    
    # Define transformations for training data
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # Define transformations for validation data
    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create BagDataset
    print("> Create BagDataset...")
    train_dataset = BagDataset(train_bags, train_labels, transform=train_transforms)
    valid_dataset = BagDataset(valid_bags, valid_labels, transform=val_transforms)
    
    print("len(train_dataset):", len(train_dataset))
    print("len(valid_dataset):", len(valid_dataset))
    
    # Create DataLoader
    print("> Create DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("len(train_loader):", len(train_loader))
    print("len(valid_loader):", len(valid_loader))

    # Initialize the model, loss function, and optimizer
    print("> Initialize the model, loss function, and optimizer...")
    if args.model == "CNN":
        instance_model = CNN()
    elif args.model == "ResNet":
        instance_model = ResNet()
    else:
        raise ValueError(f"Invalid model type: {args.model}")
    
    model = BagModel(instance_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("device:", device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join("log", args.name))
    os.makedirs(os.path.join("model", args.name), exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epoch", position=0):
        # Train the model
        # print(">> Train the model...")
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer)

        # Valid the model
        # print(">> Valid the model...")
        valid_loss, valid_accuracy = valid_model(model, valid_loader, criterion)
        
        # Write log
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_accuracy, epoch)
        writer.add_scalar('valid/loss', valid_loss, epoch)
        writer.add_scalar('valid/accuracy', valid_accuracy, epoch)
        
        print(f'\nEpoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')
        
        # Save model after each epoch
        model_save_path = os.path.join(os.path.join("model", args.name), f'epoch={epoch}.pth')
        print(f"Save model to {model_save_path}...")
        # Save the entire model
        torch.save(model, model_save_path)
    
    # Close the TensorBoard writer
    writer.close()
    
    print("> Stop training!")
