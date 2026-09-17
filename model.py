import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

NUM_EMBD = 32  # Size of the embedding vector for each token


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, num_embd):
        super().__init__()
        # Tokens are now embedded into num_embd dimensions, not straight to logits
        self.token_embedding_table = nn.Embedding(vocab_size, num_embd)
        self.position_embedding_table = nn.Embedding(block_size, num_embd)
        self.lm_head = nn.Linear(num_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tkn_emb = self.token_embedding_table(idx)  # (B,T,num_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,num_embd)
        emb = tkn_emb + pos_emb  # (B,T,num_embd)
        logits = self.lm_head(emb)  # (B,T,vocab_size)

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
