"""Saving and loading trained models."""
from dataclasses import asdict, fields
from pathlib import Path

import torch

from core.config import ROOT, Config
from core.network.model import GPT
from core.data.tokenizer import Tokenizer

CHECKPOINT_DIR = ROOT / "checkpoints"


def checkpoint_path(config):
    """Where the weights for this config's corpus live."""
    return CHECKPOINT_DIR / f"{Path(config.corpus).stem}.pt"


def save(model, tokenizer, path=None):
    """Write the weights, the config and the vocabulary to disk."""
    path = path or checkpoint_path(model.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "chars": tokenizer.chars,
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint -> {path}")


def load(config=None, path=None):
    """Rebuild the model and the tokenizer trained on config's corpus."""
    path = path or checkpoint_path(config or Config())
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
