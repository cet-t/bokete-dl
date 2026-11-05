import re
import unicodedata


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def bigrams(text):
    text = normalize(text)
    return {text[i:i + 2] for i in range(len(text) - 1)} or ({text} if text else set())


def similarity(a, b):
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x | y else 0.0
