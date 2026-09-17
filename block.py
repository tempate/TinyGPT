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
        self.proj = nn.Linear(num_embd, num_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, num_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embd, 4 * num_embd),
            nn.ReLU(),
            nn.Linear(4 * num_embd, num_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_heads, num_embd, block_size):
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            num_heads, num_embd, num_embd // num_heads, block_size
        )
        self.ffwd = FeedForward(num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
