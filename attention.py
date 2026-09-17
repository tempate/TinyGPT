import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.embd_dim, config.head_size, bias=False)
        self.query = nn.Linear(config.embd_dim, config.head_size, bias=False)
        self.value = nn.Linear(config.embd_dim, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        scores = q @ k.transpose(-2, -1) * self.config.head_size**-0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = attn @ v

        return out
