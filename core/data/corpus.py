"""Access to the training corpus.

The corpus is not kept in the repository. It is downloaded on first use into
data/, which git ignores, so no dataset is ever committed here.
"""
import re
import urllib.request
from collections import Counter
from pathlib import Path

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


def sender_counts(name):
    """How many messages each sender wrote, busiest first.

    Empty for a corpus with no "SENDER: message" structure, such as
    tiny-shakespeare.
    """
    path = DATA_DIR / name
    if not path.exists():
        return Counter()
    return Counter(re.findall(r"(?m)^([^\n:]{1,8}): ", path.read_text()))


def load_names(name):
    """Real names for the sender initials, from <corpus>.senders.txt if present.

    One "INITIAL  Name" per line. Anything parenthesised is a note, not a name.
    """
    path = DATA_DIR / f"{Path(name).stem}.senders.txt"
    if not path.exists():
        return {}

    names = {}
    for line in path.read_text().splitlines():
        initial, _, rest = line.partition(" ")
        label = rest.split(" (")[0].strip()
        if initial and label:
            names[initial] = label
    return names
