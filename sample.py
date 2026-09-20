"""Generate text from a saved checkpoint."""
import torch

from checkpoint import load


def main():
    model, tokenizer = load()

    context = torch.zeros((1, 1), dtype=torch.long, device=model.config.device)
    tokens = model.generate(context, max_new_tokens=500)
    print(tokenizer.decode(tokens[0].tolist()))


if __name__ == "__main__":
    main()
