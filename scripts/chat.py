"""Hold a conversation with a trained model."""
import argparse
import re
from collections import Counter

import torch

from core.config import Config
from core.checkpoint import load
from core.data.corpus import DATA_DIR


def detect_senders(corpus):
    """The two most talkative senders in the corpus, if it is a chat log.

    Corpora like tiny-shakespeare have no "SENDER: message" structure, in which
    case there is nobody to impersonate and the model simply continues the text.
    """
    path = DATA_DIR / corpus
    if not path.exists():
        return "", ""

    senders = re.findall(r"(?m)^([^\n:]{1,8}): ", path.read_text())
    common = [sender for sender, _ in Counter(senders).most_common(2)]
    if len(common) < 2:
        return "", ""

    bot, you = common  # The busiest sender answers; you take the runner-up
    return you, bot


def reply(model, tokenizer, transcript, max_chars):
    """Continue the transcript and return the model's next line."""
    prompt = transcript[-model.config.block_size:]
    context = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=model.config.device
    )
    tokens = model.generate(context, max_new_tokens=max_chars)
    generated = tokenizer.decode(tokens[0].tolist())[len(prompt):]
    return generated.split("\n")[0]


def drop_unknown(message, tokenizer):
    """Remove characters the corpus never contained, which have no encoding."""
    known = "".join(ch for ch in message if ch in tokenizer.stoi)
    dropped = set(message) - set(known)
    if dropped:
        print(f"  (ignoring {''.join(sorted(dropped))!r}: not in the corpus)")
    return known


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a trained TinyGPT.")
    parser.add_argument(
        "--corpus",
        default=Config.corpus,
        help="Which corpus the model was trained on (default: %(default)s)",
    )
    parser.add_argument("--you", help="The sender you speak as")
    parser.add_argument("--bot", help="The sender the model answers as")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=200,
        help="Longest reply to generate before cutting it off (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(corpus=args.corpus)
    model, tokenizer = load(config)

    you, bot = detect_senders(config.corpus)
    you = args.you or you
    bot = args.bot or bot
    you_prefix = f"{you}: " if you else ""
    bot_prefix = f"{bot}: " if bot else ""

    print(f"Trained on {config.corpus}. Ctrl-C or Ctrl-D to leave.\n")

    transcript = ""
    while True:
        try:
            message = input(you_prefix or "> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        message = drop_unknown(message.strip(), tokenizer)
        if not message:
            continue

        transcript += f"{you_prefix}{message}\n{bot_prefix}"
        line = reply(model, tokenizer, transcript, args.max_chars)
        print(f"{bot_prefix}{line}")

        # Keep only what still fits in the model's context
        transcript = (transcript + line + "\n")[-2 * model.config.block_size:]


if __name__ == "__main__":
    main()
