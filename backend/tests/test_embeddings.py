import math

from app.embeddings import HashingEmbedder


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def test_embed_returns_vector_of_configured_dimension():
    embedder = HashingEmbedder(dim=64)
    vector = embedder.embed("how do I authenticate an API request")
    assert len(vector) == 64


def test_embed_is_deterministic_across_instances():
    text = "JWT implementation for API security"
    a = HashingEmbedder(dim=128).embed(text)
    b = HashingEmbedder(dim=128).embed(text)
    assert a == b


def test_embed_output_is_l2_normalized():
    vector = HashingEmbedder(dim=128).embed("some reasonably long piece of text to embed")
    norm = math.sqrt(sum(v * v for v in vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_embed_empty_string_returns_zero_vector_without_crashing():
    vector = HashingEmbedder(dim=32).embed("")
    assert vector == [0.0] * 32


def test_similar_text_scores_higher_than_unrelated_text():
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed("How do I authenticate an API request?")
    related = embedder.embed("Authentication middleware validates API requests")
    unrelated = embedder.embed("The weather today is sunny with a light breeze")

    assert cosine(query, related) > cosine(query, unrelated)


def test_embed_batch_matches_individual_embed_calls():
    embedder = HashingEmbedder(dim=64)
    texts = ["first document", "second document", "third document"]

    batch = embedder.embed_batch(texts)
    individual = [embedder.embed(t) for t in texts]

    assert batch == individual


def test_rejects_non_positive_dimension():
    import pytest

    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
