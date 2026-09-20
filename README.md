# TinyGPT

A character-level GPT, written from scratch in PyTorch and trained on the
tiny-shakespeare corpus. Small enough to read end to end: a decoder-only
transformer in under 400 lines.

## Setup

Python 3.11+ and PyTorch are the only requirements.

```bash
git clone https://github.com/tempate/TinyGPT.git
cd TinyGPT

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m scripts.train
```

Run it from the repository root. The corpus is downloaded into `data/` on the
first run, so there is nothing to fetch by hand. Training runs for 10,000
steps, printing train and validation loss every 1,000, then writes the
weights, the config and the vocabulary to `checkpoints/checkpoint.pt` and
prints a short sample.

It picks up a GPU automatically if one is available. On CPU the default
settings are slow — see below for a smaller configuration.

## Sample

```bash
python -m scripts.sample
```

Loads `checkpoints/checkpoint.pt` and prints 500 freshly generated characters.

## Configuration

Every hyperparameter lives in `core/config.py`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `block_size` | 256 | Maximum context length |
| `embd_dim` | 384 | Embedding size |
| `num_heads` | 6 | Attention heads per block |
| `num_layers` | 6 | Transformer blocks |
| `dropout` | 0.2 | Dropout probability |
| `batch_size` | 64 | Sequences per step |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `num_steps` | 10,000 | Training steps |

To train on a CPU in a few minutes rather than hours, shrink the model:
`block_size=64`, `embd_dim=128`, `num_heads=4`, `num_layers=4`,
`num_steps=2000`.

To train on your own text, save it as `data/input.txt` before the first run
and it will be used instead of the download.

## Files

| File | Contents |
| --- | --- |
| `scripts/train.py` | Training loop and entry point |
| `scripts/sample.py` | Generation from a saved checkpoint |
| `core/network/model.py` | The `GPT` module: embeddings, blocks, language-model head |
| `core/network/block.py` | Transformer block: multi-head attention and feed-forward |
| `core/network/attention.py` | A single head of masked self-attention |
| `core/data/corpus.py` | Corpus download |
| `core/data/tokenizer.py` | Character-level encode and decode |
| `core/data/dataset.py` | Train/validation split and batch sampling |
| `core/config.py` | Hyperparameters and the repository root |
| `core/checkpoint.py` | Saving and loading trained models |

`core/network/` is the transformer itself and `core/data/` is the pipeline that
feeds it; `config` and `checkpoint` sit above both because both sides need
them. Nothing in `core/` imports anything above itself — the network never
imports the config, it receives one. `scripts/` holds the two entry points.

Generated files stay out of the source tree: the corpus lands in `data/` and
trained weights in `checkpoints/`, and git ignores each.

## Credit

Built by following Andrej Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), restructured
into separate modules. The corpus is the tiny-shakespeare dataset from
[char-rnn](https://github.com/karpathy/char-rnn).
