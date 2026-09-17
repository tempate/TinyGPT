import torch
import torch.nn as nn
from head import Head


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, num_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(num_embd, head_size, block_size) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, num_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embd, num_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
