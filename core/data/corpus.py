"""Access to the training corpus.

The corpus is not kept in the repository. It is downloaded on first use into
data/, which git ignores, so no dataset is ever committed here.
"""
import urllib.request

from core.config import ROOT

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_DIR = ROOT / "data"
SHAKESPEARE = "input.txt"  # The only corpus DATA_URL knows how to fetch


def load_text(name=SHAKESPEARE):
    """Return the named corpus from data/ as a string.

    Tiny-shakespeare is downloaded on first use. Any other corpus is expected
    to be on disk already: fetching the default URL into a file named after a
    different dataset would silently train on the wrong text.
    """
    path = DATA_DIR / name

    if not path.exists():
        if name != SHAKESPEARE:
            raise FileNotFoundError(
                f"No corpus at {path}. Only {SHAKESPEARE} is downloaded "
                f"automatically; put {name} there yourself."
            )
        print(f"Downloading {DATA_URL} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)

    return path.read_text()
