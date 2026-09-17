import torch


class Dataset:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def train_val_split(self, train_ratio):
        train_size = int(len(self.data) * train_ratio)
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        return train_data, val_data

    def split(self, train_ratio=0.9):
        """Split the dataset into training and validation sets."""
        train_data, val_data = self.train_val_split(train_ratio)
        train_dataset = Dataset(train_data, self.config)
        val_dataset = Dataset(val_data, self.config)
        return train_dataset, val_dataset

    def get_batch(self):
        """Generate a small batch of data of inputs x and targets y."""
        block_size = self.config.block_size
        ix = torch.randint(len(self.data) - block_size, (self.config.batch_size,))
        x = torch.stack([self.data[i:i + block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + block_size + 1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
