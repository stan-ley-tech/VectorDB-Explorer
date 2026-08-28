import hashlib
import math
import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic, dependency-free text embedder using the hashing trick.

    Tokens and their bigrams are hashed into a fixed-size vector (à la
    scikit-learn's HashingVectorizer): each token maps to a bucket via MD5
    (stable across processes and machines, unlike Python's salted str hash),
    and the sign of a second hash byte decides whether it adds or subtracts
    from that bucket, which keeps unrelated tokens from all piling up on the
    positive side of the same slot. The result is L2-normalized so cosine
    similarity behaves the way it would for a trained embedding model.

    This is not a substitute for a trained model's semantics, but it gives
    the rest of the system - storage, indexing, search, benchmarking - a
    real, deterministic, offline embedding source with no network calls, API
    keys, or GPU dependency. Swapping in a hosted or local model later only
    means implementing this same embed/embed_batch interface.
    """

    def __init__(self, dim: int = 256, use_bigrams: bool = True):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokenize(text)

        for token in tokens:
            _accumulate(vector, token, self.dim)

        if self.use_bigrams:
            for first, second in zip(tokens, tokens[1:]):
                _accumulate(vector, f"{first}_{second}", self.dim)

        return _l2_normalize(vector)

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _accumulate(vector: list[float], token: str, dim: int) -> None:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-12:
        return vector
    return [v / norm for v in vector]
