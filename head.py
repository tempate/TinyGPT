import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.num_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.num_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.num_embd, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.config.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)

        return out
