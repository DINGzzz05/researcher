"""Hybrid recall: embeddings + BM25 fused by RRF, optional cross-encoder rerank."""

import pytest

from gpt_researcher.memory.long_term import ResearchMemory
from gpt_researcher.memory.vectors import cosine, fuse_ranks, pack, sigmoid, unpack

REPORT_BATTERY = """# Battery recycling

## Conclusion

Battery recovery reaches 90% through direct recycling.
"""


class FakeEmbedder:
    """Deterministic stand-in: "battery" and "accumulator" share a dimension."""

    def __init__(self, name: str = "fake:mini"):
        self.name = name
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if ("battery" in lower or "accumulator" in lower) else 0.0,
                    1.0 if "cobalt" in lower else 0.0,
                    1.0 if "solar" in lower else 0.0,
                    0.05,
                ]
            )
        return vectors


class BrokenEmbedder:
    name = "broken:model"

    def embed(self, texts):
        raise RuntimeError("no credentials")


class FakeReranker:
    """Scores by a marker, so a test can force an order the fused score rejects."""

    name = "fake:reranker"

    def __init__(self, marker: str):
        self.marker = marker
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [5.0 if self.marker in text else -5.0 for text in texts]


@pytest.fixture()
def clock():
    return {"now": 1_000_000.0}


def _store(tmp_path, clock, **kwargs):
    return ResearchMemory(str(tmp_path / "m.sqlite3"), clock=lambda: clock["now"], **kwargs)


def test_vector_route_recalls_a_synonym_bm25_misses(tmp_path, clock):
    memory = _store(tmp_path, clock, embedder=FakeEmbedder())

    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )
    context = memory.recall("accumulator")

    assert context.facts
    assert "Battery recovery" in context.facts[0].text


def test_bm25_only_store_cannot_recall_the_synonym(tmp_path, clock):
    memory = _store(tmp_path, clock)

    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )

    assert memory.recall("accumulator").is_empty()


def test_rrf_prefers_an_item_found_by_both_routes():
    scores = fuse_ranks([[0, 1, 2], [1, 2, 0]])

    assert scores[0] > scores[2]
    assert scores[1] > scores[2]


def test_hybrid_keeps_both_routes(tmp_path, clock):
    memory = _store(tmp_path, clock, embedder=FakeEmbedder())
    memory.remember_run(
        "accumulator recycling",
        "Direct recycling recovers 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )
    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/b"],
    )

    context = memory.recall("accumulator")

    queries = [hit.query for hit in context.episodes]
    assert "accumulator recycling" in queries  # lexical + vector
    assert "battery recycling" in queries  # vector only
    assert context.episodes[0].query == "accumulator recycling"


def test_reranker_reorders_the_fused_candidates(tmp_path, clock):
    fused = _store(tmp_path / "fused", clock)
    fused.remember_run(
        "lithium recovery",
        "Lithium recovery reaches 70% in pilot plants.",
        report="# T\n\n## Conclusion\n\nLithium recovery reaches 70% in pilot plants.",
        sources=["https://example.com/1"],
    )
    fused.remember_run(
        "lithium recovery",
        "Lithium recovery improves with recycling; cobalt is the bottleneck.",
        report=(
            "# T\n\n## Conclusion\n\nLithium recovery improves 20% with "
            "recycling; cobalt is the bottleneck."
        ),
        sources=["https://example.com/2"],
    )

    assert "70%" in fused.recall("lithium recovery").facts[0].text

    reranked = _store(
        tmp_path / "reranked", clock, reranker=FakeReranker("bottleneck")
    )
    reranked.remember_run(
        "lithium recovery",
        "Lithium recovery reaches 70% in pilot plants.",
        report="# T\n\n## Conclusion\n\nLithium recovery reaches 70% in pilot plants.",
        sources=["https://example.com/1"],
    )
    reranked.remember_run(
        "lithium recovery",
        "Lithium recovery improves with recycling; cobalt is the bottleneck.",
        report=(
            "# T\n\n## Conclusion\n\nLithium recovery improves 20% with "
            "recycling; cobalt is the bottleneck."
        ),
        sources=["https://example.com/2"],
    )

    assert "bottleneck" in reranked.recall("lithium recovery").facts[0].text


def test_embedding_failure_falls_back_to_bm25(tmp_path, clock):
    memory = _store(tmp_path, clock, embedder=BrokenEmbedder())

    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )

    context = memory.recall("battery recycling")

    assert context.episodes
    assert memory._embedding_disabled is True


def test_stale_vectors_from_another_model_are_ignored(tmp_path, clock):
    memory = _store(tmp_path, clock, embedder=FakeEmbedder("fake:old"))
    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )

    other = _store(tmp_path, clock, embedder=FakeEmbedder("fake:new"))
    assert other.recall("accumulator").is_empty()


def test_vectors_survive_a_reopen(tmp_path, clock):
    memory = _store(tmp_path, clock, embedder=FakeEmbedder())
    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )

    reopened = _store(tmp_path, clock, embedder=FakeEmbedder())
    assert reopened.recall("accumulator").facts


def test_backfill_embeds_records_written_before_vectors(tmp_path, clock):
    memory = _store(tmp_path, clock)
    memory.remember_run(
        "battery recycling",
        "Battery recovery reaches 90%.",
        report=REPORT_BATTERY,
        sources=["https://example.com/a"],
    )
    assert memory.recall("accumulator").is_empty()

    memory.embedder = FakeEmbedder()
    assert memory.backfill_vectors() >= 1
    assert memory.recall("accumulator").facts


def test_pack_roundtrip_and_cosine():
    blob = pack([0.25, -1.0, 2.0])
    values = unpack(blob)

    assert values == pytest.approx((0.25, -1.0, 2.0))
    assert cosine((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_sigmoid_is_bounded():
    assert 0.0 < sigmoid(-8.0) < 0.01
    assert 0.99 < sigmoid(8.0) < 1.0
    assert sigmoid(0.0) == pytest.approx(0.5)
