import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split


class BagDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx] / 255.0
        label = int(self.labels[idx])
        return torch.tensor(bag, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add arguments
    parser.add_argument('--dataset_path', type=str,
                        default='./dataset', help="Path to dataset")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs")
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help="Learning rate")

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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjust based on the output size of the conv layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Output one score per instance

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BagModel(nn.Module):
    def __init__(self, instance_model):
        super(BagModel, self).__init__()
        self.instance_model = instance_model

    def forward(self, x):
        batch_size, num_instances, channels, height, width = x.size()
        # Flatten the instances into the batch dimension
        x = x.view(-1, channels, height, width)
        x = self.instance_model(x)
        # Reshape back to [batch_size, num_instances]
        x = x.view(batch_size, num_instances)
        x = torch.mean(x, dim=1)  # Aggregate the instance scores (mean)
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


def evaluate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            correct_predictions += torch.sum(preds ==
                                             labels.float().view(-1, 1))

    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct_predictions.double() / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')


if __name__ == '__main__':
    print("> Start training!")

    # Parse the arguments
    args = parse_args()

    # Load the training dataset
    print("> Load the training dataset...")
    classes = ["0", "1"]
    bags, labels = load_data(args.dataset_path, classes)

    print("len(bags):", len(bags))
    print("len(labels):", len(labels))

    # Split the dataset into training and validation sets
    train_bags, val_bags, train_labels, val_labels = train_test_split(
        bags, labels, test_size=0.2, random_state=42)

    # Create Dataset and DataLoader objects
    train_dataset = BagDataset(train_bags, train_labels)
    val_dataset = BagDataset(val_bags, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the instance model and the bag model
    instance_model = SimpleCNN()
    model = BagModel(instance_model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, args.epochs)

    # Evaluate the model
    evaluate_model(model, val_loader, criterion)
