import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BagDataset(Dataset):
    def __init__(self, bags, labels, transform=None):
        self.bags = bags
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx] / 255.0
        label = int(self.labels[idx])
        
        # bag = Image.fromarray((bag).astype(np.uint8))
        # if self.transform:
        #     bag = self.transform(bag)
        
        # Reshape to [channels, depth, height, width]
        return torch.tensor(bag, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(label, dtype=torch.long)
