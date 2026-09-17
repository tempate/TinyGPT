import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

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
        for _ in range(max_new_tokens):
            # Get the last prediction
            logits, _ = self(context)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution or take the most likely
            next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            context = torch.cat((context, next), dim=1)

        return context
