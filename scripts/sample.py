"""Generate text from a saved checkpoint."""
import argparse

import torch

from core.checkpoint import load


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained TinyGPT.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Below 1 is more predictable, above 1 more chaotic (default: %(default)s)",
    )
    parser.add_argument(
        "--chars", type=int, default=500, help="How much to generate (default: %(default)s)"
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    return args


def main():
    args = parse_args()
    model, tokenizer = load()

    context = torch.zeros((1, 1), dtype=torch.long, device=model.config.device)
    tokens = model.generate(context, max_new_tokens=args.chars, temperature=args.temperature)
    print(tokenizer.decode(tokens[0].tolist()))


if __name__ == "__main__":
    main()
