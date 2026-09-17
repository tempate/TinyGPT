import torch
import torch.nn as nn
import torch.nn.functional as F
from block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.num_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.num_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.num_embd)
        self.lm_head = nn.Linear(config.num_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tkn_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tkn_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        # Reshape logits and targets to compute cross-entropy loss
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens):
        """Generate new tokens from a given context."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop the model input to block size
            context_window = context[:, -self.config.block_size:]

            # Get the last prediction
            logits, _ = self(context_window)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution or take the most likely
            next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            context = torch.cat((context, next), dim=1)

        self.train()
        return context
