import torch


class Dataset:
    def __init__(self, data, device, block_size):
        self.data = data
        self.device = device
        self.block_size = block_size

    def train_val_split(self, train_ratio):
        train_size = int(len(self.data) * train_ratio)
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        return train_data, val_data

    def split(self, train_ratio=0.9):
        """Split the dataset into training and validation sets."""
        train_data, val_data = self.train_val_split(train_ratio)
        train_dataset = Dataset(train_data, self.device, self.block_size)
        val_dataset = Dataset(val_data, self.device, self.block_size)
        return train_dataset, val_dataset

    def get_batch(self, batch_size):
        """Generate a small batch of data of inputs x and targets y."""
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
