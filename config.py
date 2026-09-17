from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # Model
    block_size: int = 8  # Maximum context length the model can attend to
    num_embd: int = 32  # Size of the embedding vector for each token
    vocab_size: int = None  # Filled in once the tokenizer has seen the text
    head_size: int = 8  # Size of each attention head
    num_heads: int = 4  # Number of attention heads
    num_layers: int = 3  # Number of transformer layers
    dropout: float = 0.2  # Dropout probability

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_steps: int = 10_000
    eval_interval: int = 1_000
    eval_iters: int = 200

    # Runtime
    seed: int = 1337
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu'
    )
