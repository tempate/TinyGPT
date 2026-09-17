from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # Model
    block_size: int = 256  # Maximum context length the model can attend to
    num_embd: int = 384  # Size of the embedding vector for each token
    vocab_size: int = None  # Filled in once the tokenizer has seen the text
    head_size: int = field(init=False)  # Size of each attention head
    num_heads: int = 6  # Number of attention heads
    num_layers: int = 6  # Number of transformer layers
    dropout: float = 0.2  # Dropout probability

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_steps: int = 10_000
    eval_interval: int = 1_000
    eval_iters: int = 200

    # Runtime
    seed: int = 1337
    device: str = field(init=False)

    def __post_init__(self):
        self.head_size = self.num_embd // self.num_heads
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
