import torch
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = torch.tensor(self.bags[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.int)
        
        return bag, label
