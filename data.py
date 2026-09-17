"""Access to the training corpus.

The corpus is not kept in the repository. It is downloaded on first use and
ignored by git, so no dataset is ever committed here.
"""
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent / "input.txt"


def load_text(path=DATA_PATH, url=DATA_URL):
    """Return the corpus as a string, downloading it if it is not on disk."""
    if not path.exists():
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path.read_text()
