from core.config import Config
from core.model import GPT
from core.tokenizer import Tokenizer
from core.dataset import Dataset
from core.checkpoint import save
from core.corpus import load_text

import torch


def train(config, model, train_dataset, val_dataset):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for step in range(config.num_steps):
        x, y = train_dataset.get_batch()

        if step % config.eval_interval == 0:
            # Evaluate loss on train and val sets
            train_loss = estimate_loss(config, model, train_dataset)
            val_loss = estimate_loss(config, model, val_dataset)
            print(f"Step {step}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

        # Forward pass
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")


@torch.no_grad()
def estimate_loss(config, model, dataset):
    """Estimate the loss on the given data."""
    model.eval()

    losses = torch.zeros(config.eval_iters)
    for i in range(config.eval_iters):
        x, y = dataset.get_batch()
        _, loss = model(x, y)
        losses[i] = loss.item()

    model.train()
    return losses.mean()


def main():
    config = Config()
    torch.manual_seed(config.seed)

    # Read the dataset, downloading it on first run
    text = load_text()

    # Tokenize the dataset
    tokenizer = Tokenizer.from_text(text)
    config.vocab_size = tokenizer.vocab_size

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_dataset, val_dataset = Dataset(data, config).split()

    model = GPT(config).to(config.device)
    train(config, model, train_dataset, val_dataset)
    save(model, tokenizer)

    tokens = torch.zeros((1, 1), dtype=torch.long, device=config.device)  # Starting token (e.g., BOS token)
    new_tokens = model.generate(tokens, max_new_tokens=100)
    print(tokenizer.decode(new_tokens[0].tolist()))


if __name__ == "__main__":
    main()
