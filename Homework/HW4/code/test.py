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
import pandas as pd

from dataset import BagDataset
from model.Bag import BagMean, BagMax
from model.CNN import CNN
from model.ResNet import ResNet18, ResNet34, ResNet50


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training arguments")

    # Add arguments
    parser.add_argument('--name', type=str, default="test", help="Model name", required=True)
    parser.add_argument('--model', type=str, default="test", help="Model type")
    parser.add_argument('--dataset_path', type=str, default='./dataset', help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.00001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization weight decay")

    # Parse the arguments
    args = parser.parse_args()
    
    return args


def load_data(dataset_path):
    bags = []
    labels = []
    ids = []

    test_dataset_path = os.path.join(dataset_path, "test")
    test_dataset_files = os.listdir(test_dataset_path)
    print(f">> Load the {test_dataset_path}...")

    # For each file
    for test_dataset_file in test_dataset_files:
        test_dataset_file_path = os.path.join(
            test_dataset_path, test_dataset_file)

        # Load the file
        with open(test_dataset_file_path, 'rb') as f:
            data = pickle.load(f)

        bags.append(data)
        labels.append(0)
        ids.append(test_dataset_file.split(".")[0])

    return bags, labels, ids


def test_model(model, test_loader, criterion):
    model.eval()
    test_preds = []
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test", position=0):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
            test_preds += preds.cpu()

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = correct_predictions / total_samples
    
    test_preds_int = [int(preds.item()) for preds in test_preds]

    return test_loss, test_accuracy, test_preds_int


if __name__ == '__main__':
    print("> Start testing!")
    
    # Parse the arguments
    args = parse_args()
    
    # Load the testing dataset
    print("> Load the testing dataset...")
    test_bags, test_labels, test_ids = load_data(args.dataset_path)
            
    # print("len(bags):", len(bags))
    # print("len(labels):", len(labels))
    
    # Define transformations for validation data
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create BagDataset
    print("> Create BagDataset...")
    test_dataset = BagDataset(test_bags, test_labels, transform=test_transforms)
    
    print("len(test_dataset):", len(test_dataset))
    
    # Create DataLoader
    print("> Create DataLoader...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("len(test_loader):", len(test_loader))

    # Initialize the model, loss function, and optimizer
    print("> Initialize the model, loss function, and optimizer...")
    # Load the model
    model = torch.load(f"./model/{args.name}.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("device:", device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Test the model
    print("> Test the model...")
    test_loss, test_accuracy, test_preds = test_model(model, test_loader, criterion)
    
    # Output submission file
    print("> Output submission file...")
    output = {
        "image_id": test_ids,
        "y_pred": test_preds
    }
    # Create a DataFrame
    df = pd.DataFrame(output)
    # Write DataFrame to CSV file
    os.makedirs(os.path.join("submission", args.name.split("/")[0]), exist_ok=True)
    df.to_csv(f"./submission/{args.name}.csv", index=False)
    
    print("> Stop testing!")
