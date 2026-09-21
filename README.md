# TinyGPT

A character-level GPT written from scratch in PyTorch, under 400 lines.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Run from the repository root:

```bash
python -m scripts.train                        # tiny-shakespeare, downloaded on first run
python -m scripts.train --corpus chats.txt     # any other file you put in data/
```

Weights are written to `checkpoints/`, named after the corpus. A GPU is used if
there is one.

## Sample

```bash
python -m scripts.sample
```

Prints 500 characters from the trained model. `--temperature` below 1 makes it
more predictable, above 1 more chaotic.

## Chat

```bash
python -m scripts.chat --corpus chats.txt
```

Type a message and the model answers. If the corpus looks like a chat log it
picks the two busiest senders to speak as; override with `--you` and `--bot`.
`--temperature` works here too.

## Layout

```
core/network/        attention.py, block.py, model.py
core/data/           corpus.py, tokenizer.py, dataset.py
core/config.py       every hyperparameter
core/checkpoint.py   saving and loading
scripts/             train.py, sample.py, chat.py
data/                corpora      (git-ignored)
checkpoints/         weights      (git-ignored)
```

## Configuration

Everything is in `core/config.py`. The defaults assume a GPU; on a CPU use
`block_size=64`, `embd_dim=128`, `num_heads=4`, `num_layers=4`,
`num_steps=2000` to train in a few minutes.

## Credit

Follows Andrej Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). The default
corpus is tiny-shakespeare from
[char-rnn](https://github.com/karpathy/char-rnn).
