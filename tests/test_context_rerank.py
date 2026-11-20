"""Context reranking: config wiring, failure fallback and final ordering."""

from types import SimpleNamespace

from langchain_core.documents import Document

from gpt_researcher.context import select as select_module
from gpt_researcher.context.rerank import (
    context_rerank_model,
    context_rerank_top_n,
)
from gpt_researcher.prompts import PromptFamily


def doc(text: str) -> Document:
    return Document(page_content=text, metadata={"title": text, "source": f"https://example.com/{text}"})


def _cfg(**overrides):
    base = {
        "context_token_budget": 0,
        "smart_llm_model": None,
        "context_order": "relevance",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------- config


def test_rerank_model_comes_from_config(monkeypatch):
    monkeypatch.delenv("RERANK_MODEL", raising=False)

    cfg = SimpleNamespace(context_rerank="cross-encoder:BAAI/bge-reranker-base")

    assert context_rerank_model(cfg) == "BAAI/bge-reranker-base"


def test_rerank_off_ignores_the_legacy_env(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "legacy-model")

    assert context_rerank_model(SimpleNamespace(context_rerank="off")) is None


def test_rerank_model_falls_back_to_the_legacy_env(monkeypatch):
    monkeypatch.setenv("RERANK_MODEL", "legacy-model")

    assert context_rerank_model(SimpleNamespace(context_rerank="")) == "legacy-model"


def test_rerank_top_n_comes_from_config(monkeypatch):
    monkeypatch.delenv("RERANK_TOP_N", raising=False)

    assert context_rerank_top_n(SimpleNamespace(context_rerank_top_n=12)) == 12
    assert context_rerank_top_n(SimpleNamespace(context_rerank_top_n=0)) is None


def test_rerank_top_n_falls_back_to_the_legacy_env(monkeypatch):
    monkeypatch.setenv("RERANK_TOP_N", "7")

    assert context_rerank_top_n(SimpleNamespace()) == 7


# ------------------------------------------------------------- hybrid rerank


def _fixed_lexical(monkeypatch, docs):
    monkeypatch.setattr(
        select_module,
        "LexicalContextCompressor",
        lambda **_: type("Lex", (), {"select": lambda self, query, limit: docs})(),
    )


async def test_hybrid_select_reranks_with_the_configured_model(monkeypatch):
    docs = [doc("A"), doc("B"), doc("C")]
    _fixed_lexical(monkeypatch, docs)
    monkeypatch.setattr(
        "gpt_researcher.utils.rerank.load_scorer",
        lambda model: (lambda pairs: [0.1, 0.9, 0.5]),
    )
    cfg = SimpleNamespace(context_rerank="fake-model", context_rerank_top_n=10)

    result = await select_module.hybrid_select("q", [{"raw_content": "x"}], cfg)

    assert [d.page_content for d in result] == ["B", "C", "A"]


async def test_hybrid_select_keeps_fused_order_when_the_model_fails(monkeypatch):
    docs = [doc("A"), doc("B"), doc("C")]
    _fixed_lexical(monkeypatch, docs)

    def broken(model):
        raise RuntimeError("no weights")

    monkeypatch.setattr("gpt_researcher.utils.rerank.load_scorer", broken)
    cfg = SimpleNamespace(context_rerank="fake-model", context_rerank_top_n=10)

    result = await select_module.hybrid_select("q", [{"raw_content": "x"}], cfg)

    assert [d.page_content for d in result] == ["A", "B", "C"]


# ------------------------------------------------------------------ ordering


def test_lost_in_the_middle_puts_the_best_passages_at_the_edges():
    docs = [doc("AAA"), doc("BBB"), doc("CCC")]

    text = select_module._render(
        docs, None, _cfg(context_order="lost_in_middle"), PromptFamily
    )

    assert text.index("AAA") < text.index("CCC") < text.index("BBB")


def test_default_order_stays_relevance():
    docs = [doc("AAA"), doc("BBB"), doc("CCC")]

    text = select_module._render(docs, None, _cfg(), PromptFamily)

    assert text.index("AAA") < text.index("BBB") < text.index("CCC")


def test_lost_in_the_middle_leaves_two_passages_alone():
    docs = [doc("AAA"), doc("BBB")]

    text = select_module._render(
        docs, None, _cfg(context_order="lost_in_middle"), PromptFamily
    )

    assert text.index("AAA") < text.index("BBB")


def test_unknown_order_falls_back_to_relevance():
    docs = [doc("AAA"), doc("BBB"), doc("CCC")]

    text = select_module._render(docs, None, _cfg(context_order="bogus"), PromptFamily)

    assert text.index("AAA") < text.index("BBB") < text.index("CCC")
