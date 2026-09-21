"""Export a checkpoint as a flat binary the browser can load.

Weights go out as one float32 blob in a fixed order, with a small JSON manifest
naming the shapes and the vocabulary. The attention masks are left behind: they
are constant, so the browser rebuilds them instead of downloading them.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from core.config import Config
from core.checkpoint import checkpoint_path
from core.data.corpus import load_names, sender_counts
from scripts.chat import detect_senders


def tensor_order(num_layers, num_heads):
    """Every weight the browser needs, in the order it is written."""
    names = ["token_embedding_table.weight", "position_embedding_table.weight"]
    for i in range(num_layers):
        block = f"blocks.{i}"
        names += [f"{block}.ln1.weight", f"{block}.ln1.bias"]
        names += [
            f"{block}.attn.heads.{h}.{proj}.weight"
            for h in range(num_heads)
            for proj in ("key", "query", "value")
        ]
        names += [f"{block}.attn.proj.weight", f"{block}.attn.proj.bias"]
        names += [f"{block}.ln2.weight", f"{block}.ln2.bias"]
        names += [
            f"{block}.ffwd.net.0.weight", f"{block}.ffwd.net.0.bias",
            f"{block}.ffwd.net.2.weight", f"{block}.ffwd.net.2.bias",
        ]
    names += ["ln_f.weight", "ln_f.bias", "lm_head.weight", "lm_head.bias"]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=Config.corpus, help="Which checkpoint to export")
    parser.add_argument("--out", default="web", help="Directory to write into")
    args = parser.parse_args()

    checkpoint = torch.load(checkpoint_path(Config(corpus=args.corpus)), map_location="cpu")
    state, config = checkpoint["state_dict"], checkpoint["config"]

    # Each head is its own module, so the count comes from the config
    names = tensor_order(config["num_layers"], config["num_heads"])
    missing = [n for n in names if n not in state]
    extra = [n for n in state if n not in names and not n.endswith(".tril")]
    assert not extra, f"not exported: {extra[:3]}"
    assert not missing, f"missing from checkpoint: {missing[:3]}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    blob, manifest = bytearray(), []
    for name in names:
        array = state[name].numpy().astype(np.float32)
        manifest.append({"name": name, "shape": list(array.shape)})
        blob += array.tobytes(order="C")

    (out / "weights.bin").write_bytes(bytes(blob))
    # The browser has no corpus to scan, so the speakers ship with the model:
    # every sender in the order they talk, their real names where we know them,
    # and the two the page should start with.
    you, bot = detect_senders(args.corpus)
    sender_names = load_names(args.corpus)
    speakers = [
        {"id": sender, "name": sender_names.get(sender, sender), "messages": count}
        for sender, count in sender_counts(args.corpus).most_common()
    ]

    (out / "model.json").write_text(json.dumps({
        "config": {k: config[k] for k in
                   ("block_size", "embd_dim", "num_heads", "num_layers", "head_size", "vocab_size")},
        "chars": checkpoint["chars"],
        "senders": {"you": you, "bot": bot},
        "speakers": speakers,
        "corpus": args.corpus,
        "tensors": manifest,
    }))

    print(f"{len(names)} tensors, {len(blob)/1e6:.2f} MB -> {out}/weights.bin")
    print(f"vocab {len(checkpoint['chars'])}, {len(speakers)} speakers, "
          f"defaults {you!r}/{bot!r} -> {out}/model.json")


if __name__ == "__main__":
    main()
