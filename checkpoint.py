"""Saving and loading trained models."""
from dataclasses import asdict, fields
from pathlib import Path

import torch

from config import Config
from model import GPT
from tokenizer import Tokenizer

CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.pt"


def save(model, tokenizer, path=CHECKPOINT_PATH):
    """Write the weights, the config and the vocabulary to disk."""
    torch.save(
        {
            "config": asdict(model.config),
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint -> {path}")


def load(path=CHECKPOINT_PATH):
    """Rebuild the model and the tokenizer stored at path."""
    checkpoint = torch.load(path, map_location="cpu")

    config = config_from_dict(checkpoint["config"])
    tokenizer = Tokenizer(checkpoint["chars"])

    model = GPT(config).to(config.device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, tokenizer


def config_from_dict(values):
    """Rebuild a Config, skipping the fields __post_init__ derives."""
    supplied = {field.name for field in fields(Config) if field.init}
    return Config(**{k: v for k, v in values.items() if k in supplied})
