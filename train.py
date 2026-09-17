from model import BigramLanguageModel
from tokenizer import Tokenizer
from dataset import Dataset
from data import load_text

import torch


SEED = 1337
torch.manual_seed(SEED)
BATCH_SIZE = 32
BLOCK_SIZE = 8
LEARNING_RATE = 1e-3
NUM_STEPS = 10_000


def train(model, train_dataset, val_dataset):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(NUM_STEPS):
        x, y = train_dataset.get_batch(BATCH_SIZE)

        if step % 1000 == 0:
            # Evaluate loss on train and val sets
            train_loss = estimate_loss(model, train_dataset, eval_iters=200)
            val_loss = estimate_loss(model, val_dataset, eval_iters=200)
            print(f"Step {step}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

        # Forward pass
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters=200):
    """Estimate the loss on the given data."""
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = dataset.get_batch(BATCH_SIZE)
        _, loss = model(x, y)
        losses[i] = loss.item()
    return losses.mean()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read the dataset, downloading it on first run
    text = load_text()

    # Tokenize the dataset
    tokenizer = Tokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = Dataset(data, device, block_size=BLOCK_SIZE).split()

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    train(model, train_dataset, val_dataset)

    tokens = torch.zeros((1, 1), dtype=torch.long, device=device)  # Starting token (e.g., BOS token)
    new_tokens = model.generate(tokens, max_new_tokens=100)
    print(tokenizer.decode(new_tokens[0].tolist()))


if __name__ == "__main__":
    main()
