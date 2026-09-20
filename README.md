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
python train.py
```

The corpus is downloaded on the first run, so there is nothing to fetch by
hand. Training runs for 10,000 steps, printing train and validation loss every
1,000, then writes the weights, the config and the vocabulary to
`checkpoint.pt` and prints a short sample.

It picks up a GPU automatically if one is available. On CPU the default
settings are slow — see below for a smaller configuration.

## Sample

```bash
python sample.py
```

Loads `checkpoint.pt` and prints 500 freshly generated characters.

## Configuration

Every hyperparameter lives in `config.py`:

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

To train on your own text, drop it in as `input.txt` before the first run and
it will be used instead of the download.

## Files

| File | Contents |
| --- | --- |
| `train.py` | Training loop and entry point |
| `sample.py` | Generation from a saved checkpoint |
| `model.py` | The `GPT` module: embeddings, blocks, language-model head |
| `block.py` | Transformer block: multi-head attention and feed-forward |
| `attention.py` | A single head of masked self-attention |
| `config.py` | Hyperparameters |
| `tokenizer.py` | Character-level encode and decode |
| `dataset.py` | Train/validation split and batch sampling |
| `data.py` | Corpus download |
| `checkpoint.py` | Saving and loading trained models |

## Credit

Built by following Andrej Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), restructured
into separate modules. The corpus is the tiny-shakespeare dataset from
[char-rnn](https://github.com/karpathy/char-rnn).
