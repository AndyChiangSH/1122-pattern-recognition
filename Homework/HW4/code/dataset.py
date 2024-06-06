import torch
from torch.utils.data import Dataset


class BagDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx] / 255.0
        label = int(self.labels[idx])
        # Reshape to [channels, depth, height, width]
        return torch.tensor(bag, dtype=torch.float32).permute(0, 3, 1, 2), torch.tensor(label, dtype=torch.long)
