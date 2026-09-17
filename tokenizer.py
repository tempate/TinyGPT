

class Tokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
