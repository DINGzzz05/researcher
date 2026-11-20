"""Three-layer research memory.

The first version of this module stored one thing twice: a run's query and
summary, once with a 24 h TTL and once with a 30 d TTL. Two tiers, one kind of
content - so "long-term memory" was only a slower cache of the session one.

This version stores three *kinds* of content, each with its own lifecycle and
its own job at retrieval time:

* **episodic** (``episodes``) - one row per finished run: query, summary,
  sources, entities, session. Answers "have we researched this before, and
  what came out of it?". TTL-bounded, recency-weighted.
  (MemGPT's recall memory; Graphiti's episodic nodes.)
* **semantic** (``facts``) - the claims distilled out of those reports,
  deduped and merged across runs, carrying confidence, mention count and a
  validity window. Answers "what do we already know about X?". Outlives the
  episode that first reported it.
  (Mem0's extraction + ADD/UPDATE/NOOP; Graphiti's temporal fact
  invalidation.)
* **procedural** (``profile``) - the user's recurring topics, decayed as new
  runs arrive. Answers "what does this user keep working on?" and steers
  planning rather than recall.
  (LangMem's procedural memory.)

The session is a scope, not a tier: rows written by the current session are
recalled first as a bounded working set, like MemGPT's core memory block; the
rest of the user's history competes on score.

Recall blends relevance (BM25, plus an embedding route when one is
configured), recency (a half-life per layer), importance (fact confidence,
episode source count) and entity overlap - the relevance/recency/importance
blend from Generative Agents, minus the model. With ``MEMORY_RERANK`` set, the
fused candidates are reordered by a local cross-encoder before that blend
decides the final order. Consolidation runs at write time: token Jaccard
decides whether a new claim merges into an existing fact, supersedes it (same
claim, different number), or is added. No embeddings or model are required;
both are optional routes that degrade to BM25.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterable

from ..context.dedup import jaccard
from ..context.lexical import bm25_scores, tokenize
from .extract import extract_entities, extract_facts
from .rerank import build_reranker
from .vectors import build_embedder, cosine, fuse_ranks, pack, sigmoid, unpack

logger = logging.getLogger(__name__)

#: Keyword count kept per record, used for the preference profile.
KEYWORD_LIMIT = 8
#: Days after which relevance halves, per layer.
RECENCY_HALF_LIFE_DAYS = {"episode": 7.0, "fact": 30.0}
#: Recall weight added to episodes from the current session.
SESSION_BONUS = 0.35
#: Jaccard similarity at which a claim is the same fact (merge).
MERGE_SIMILARITY = 0.75
#: Jaccard similarity at which a claim is about the same thing (supersede).
SUPERSEDE_SIMILARITY = 0.5
#: Confidence of a fact seen once, and the ceiling repeated sightings approach.
BASE_CONFIDENCE = 0.55
MAX_CONFIDENCE = 0.98
#: Weight decay applied to a profile term each time it is seen again.
PROFILE_DECAY = 0.98
#: Profile terms kept per user.
PROFILE_TERMS = 300
#: Bounded working set injected from the current session.
SESSION_HITS = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    query TEXT NOT NULL,
    summary TEXT NOT NULL,
    sources TEXT NOT NULL,
    keywords TEXT NOT NULL,
    entities TEXT NOT NULL,
    vector BLOB,
    vector_model TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_expiry ON episodes(expires_at);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    subject TEXT NOT NULL,
    tokens TEXT NOT NULL,
    sources TEXT NOT NULL,
    entities TEXT NOT NULL,
    confidence REAL NOT NULL,
    times_seen INTEGER NOT NULL,
    episode_id TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    valid_to REAL,
    superseded_by TEXT,
    expires_at REAL NOT NULL,
    vector BLOB,
    vector_model TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_expiry ON facts(expires_at);

CREATE TABLE IF NOT EXISTS profile (
    user_id TEXT NOT NULL,
    term TEXT NOT NULL,
    weight REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, term)
);
"""


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Episode:
    """One remembered research run."""

    id: str
    user_id: str
    session_id: str | None
    query: str
    summary: str
    sources: tuple[str, ...]
    keywords: tuple[str, ...]
    entities: tuple[str, ...]
    created_at: float
    expires_at: float
    vector: tuple[float, ...] = ()
    vector_model: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "summary": self.summary,
            "sources": list(self.sources),
            "keywords": list(self.keywords),
            "entities": list(self.entities),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class Fact:
    """One claim distilled out of one or more runs."""

    id: str
    user_id: str
    claim: str
    subject: str
    tokens: tuple[str, ...]
    sources: tuple[str, ...]
    entities: tuple[str, ...]
    confidence: float
    times_seen: int
    episode_id: str | None
    first_seen: float
    last_seen: float
    valid_to: float | None
    superseded_by: str | None
    expires_at: float
    vector: tuple[float, ...] = ()
    vector_model: str = ""

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


@dataclass(frozen=True)
class MemoryHit:
    """One recalled item, from any layer, with the score that ranked it."""

    layer: str  # "session" | "episode" | "fact"
    id: str
    text: str
    score: float
    age_days: float
    query: str = ""
    subject: str = ""
    confidence: float | None = None
    times_seen: int = 1
    sources: tuple[str, ...] = ()
    relevance: float = 0.0
    quality: float = 0.0
    entity: float = 0.0
    matched: bool = False


@dataclass
class MemoryContext:
    """Everything recall found, already split by layer for the prompt."""

    session: list[MemoryHit] = field(default_factory=list)
    episodes: list[MemoryHit] = field(default_factory=list)
    facts: list[MemoryHit] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.session or self.episodes or self.facts)

    def to_dict(self) -> dict:
        return {
            "session": [hit.__dict__ for hit in self.session],
            "episodes": [hit.__dict__ for hit in self.episodes],
            "facts": [hit.__dict__ for hit in self.facts],
            "profile": list(self.profile),
        }


@dataclass(frozen=True)
class FactOutcome:
    """What consolidation did with one extracted claim."""

    action: str  # "added" | "merged" | "superseded"
    fact_id: str
    replaced: str | None = None


@dataclass(frozen=True)
class RunWrite:
    """What one finished run wrote into memory."""

    episode: Episode | None
    facts: tuple[FactOutcome, ...]


def extract_keywords(text: str, limit: int = KEYWORD_LIMIT) -> list[str]:
    """Most frequent meaningful terms in ``text`` (stopwords removed, stemmed)."""
    counts = Counter(tokenize(text or ""))
    return [term for term, _ in counts.most_common(limit)]


def _numbers(text: str) -> frozenset[str]:
    """Numbers in a claim, used to notice "70%" becoming "80%"."""
    return frozenset(re.findall(r"\d+(?:[.,]\d+)?", text or ""))


# --------------------------------------------------------------------------- store


class ResearchMemory:
    """SQLite-backed episodic + semantic + procedural memory.

    ``clock`` is injectable so expiry and decay can be tested without sleeping.
    """

    def __init__(
        self,
        path: str,
        *,
        episode_ttl_days: float = 30,
        fact_ttl_days: float = 365,
        session_hours: float = 24,
        max_episodes: int = 500,
        max_facts: int = 1_000,
        facts_per_run: int = 6,
        embedder=None,
        reranker=None,
        rerank_candidates: int = 24,
        vector_min_cos: float = 0.3,
        clock=time.time,
    ):
        if not path:
            raise ValueError("ResearchMemory needs a database path")
        self.path = path
        self.episode_ttl = float(episode_ttl_days) * 86_400
        self.fact_ttl = float(fact_ttl_days) * 86_400
        self.session_hours = float(session_hours)
        self.max_episodes = max(1, int(max_episodes))
        self.max_facts = max(1, int(max_facts))
        self.facts_per_run = max(0, int(facts_per_run))
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_candidates = max(2, int(rerank_candidates or 2))
        self.vector_min_cos = float(vector_min_cos)
        self._embedding_disabled = False
        self.clock = clock
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._init_db()

    # ---------------------------------------------------------------- storage

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._ensure_vector_columns()
        self._migrate_legacy()

    def _ensure_vector_columns(self) -> None:
        """Add embedding columns to databases created before hybrid recall."""
        with self._connect() as conn:
            for table in ("episodes", "facts"):
                columns = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table})")
                }
                if "vector" not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN vector BLOB")
                if "vector_model" not in columns:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN vector_model TEXT NOT NULL DEFAULT ''"
                    )

    def _migrate_legacy(self) -> None:
        """Carry a v1 ``memories`` table over, then park it under a new name.

        v1 wrote every run twice (session copy + long copy). Only the long copy
        is imported: the session copy was the same row with a shorter TTL.
        """
        rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            tables = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if "memories" not in tables:
                return
            try:
                rows = conn.execute(
                    "SELECT user_id, session_id, query, summary, sources, created_at "
                    "FROM memories WHERE tier = 'long'"
                ).fetchall()
                conn.execute("ALTER TABLE memories RENAME TO memories_legacy_v1")
            except sqlite3.DatabaseError:
                return

        for row in rows:
            try:
                self.remember_run(
                    row["query"],
                    row["summary"],
                    report=row["summary"],
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    sources=json.loads(row["sources"]),
                )
            except (sqlite3.DatabaseError, ValueError):
                continue

    def remember_run(
        self,
        query: str,
        summary: str,
        *,
        report: str = "",
        user_id: str = "default",
        session_id: str | None = None,
        sources: Iterable[str] | None = None,
        facts: Iterable[str] | None = None,
    ) -> RunWrite:
        """Store a finished run: one episode, its claims, and the profile bump."""
        query = (query or "").strip()
        summary = (summary or "").strip()
        if not query and not summary:
            return RunWrite(episode=None, facts=())

        user = user_id or "default"
        now = self.clock()
        keywords = extract_keywords(f"{query}\n{summary}")
        entities = extract_entities(f"{query}\n{summary}", extra=keywords[:3])
        claims = (
            list(facts)
            if facts is not None
            else extract_facts(report or summary, query, limit=self.facts_per_run)
        )

        # One batch: the episode text and every claim it produced. When the
        # embedder is unavailable this is a no-op and vectors stay empty, so
        # recall falls back to BM25 alone.
        vectors = self._embed([f"{query}\n{summary}", *claims])
        episode_vector = vectors[0] if vectors else ()
        claim_vectors = vectors[1:] if vectors else [() for _ in claims]

        episode = Episode(
            id=uuid.uuid4().hex,
            user_id=user,
            session_id=session_id,
            query=query,
            summary=summary,
            sources=tuple(source for source in (sources or ()) if source),
            keywords=tuple(keywords),
            entities=tuple(entities),
            created_at=now,
            expires_at=now + self.episode_ttl,
            vector=episode_vector,
            vector_model=self._embedder_name if episode_vector else "",
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO episodes (id, user_id, session_id, query, summary, sources, "
                "keywords, entities, vector, vector_model, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode.id, episode.user_id, episode.session_id, episode.query,
                    episode.summary, json.dumps(list(episode.sources)),
                    json.dumps(list(episode.keywords)), json.dumps(list(episode.entities)),
                    pack(episode.vector) if episode.vector else None,
                    episode.vector_model,
                    episode.created_at, episode.expires_at,
                ),
            )

        outcomes = tuple(
            outcome
            for outcome in (
                self._consolidate(claim, episode, now, vector=vector)
                for claim, vector in zip(claims, claim_vectors)
            )
            if outcome is not None
        )

        # The profile is what the *user* keeps asking about, so it is keyed on
        # the query, not on each report's body text.
        topic_terms = extract_keywords(query)
        self._update_profile(
            user,
            [*topic_terms, *extract_entities(query, extra=topic_terms[:3])],
            now,
        )
        self._prune_episodes(user)
        self._prune_facts(user)
        try:
            self.forget_expired()
        except sqlite3.DatabaseError:
            pass
        return RunWrite(episode=episode, facts=outcomes)

    def _consolidate(
        self, claim: str, episode: Episode, now: float, vector: tuple[float, ...] = ()
    ) -> FactOutcome | None:
        """Add, merge or supersede one claim against the user's active facts."""
        claim = (claim or "").strip()
        if not claim:
            return None

        tokens = tuple(dict.fromkeys(tokenize(claim)))
        token_set = frozenset(tokens)
        numbers = _numbers(claim)
        entities = extract_entities(claim, extra=episode.entities[:3])
        confidence = min(
            0.85, BASE_CONFIDENCE + 0.1 * min(3, len(episode.sources))
        )
        vector_model = self._embedder_name if vector else ""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE user_id = ? AND valid_to IS NULL AND expires_at > ?",
                (episode.user_id, now),
            ).fetchall()

            best, best_similarity, best_numbers = None, 0.0, frozenset()
            for row in rows:
                similarity = jaccard(token_set, frozenset(json.loads(row["tokens"])))
                if similarity > best_similarity:
                    best, best_similarity = row, similarity
                    best_numbers = _numbers(row["claim"])

            subject = _subject_for(claim, entities, episode)
            if best is not None and best_similarity >= MERGE_SIMILARITY and not _conflicts(
                best_numbers, numbers
            ):
                merged_sources = sorted(
                    set(json.loads(best["sources"])).union(episode.sources)
                )
                merged_entities = list(
                    dict.fromkeys([*json.loads(best["entities"]), *entities])
                )
                # Keep the more informative wording: "costs are high" merged
                # with "costs are 30% higher in the EU" should store the latter.
                previous_tokens = json.loads(best["tokens"])
                richer = len(tokens) > len(previous_tokens) + 2
                stored_vector = best["vector"]
                if richer and vector:
                    # The stored claim changed, so its vector must change with it.
                    stored_vector = pack(vector)
                conn.execute(
                    "UPDATE facts SET claim = ?, subject = ?, tokens = ?, confidence = ?, "
                    "times_seen = ?, last_seen = ?, sources = ?, entities = ?, "
                    "vector = ?, vector_model = ?, expires_at = ? WHERE id = ?",
                    (
                        claim if richer else best["claim"],
                        subject if richer else best["subject"],
                        json.dumps(list(tokens) if richer else previous_tokens),
                        min(MAX_CONFIDENCE, float(best["confidence"]) + 0.1),
                        int(best["times_seen"]) + 1,
                        now,
                        json.dumps(merged_sources),
                        json.dumps(merged_entities),
                        stored_vector,
                        vector_model if richer and vector else best["vector_model"],
                        now + self.fact_ttl,
                        best["id"],
                    ),
                )
                return FactOutcome("merged", best["id"])

            fact_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO facts (id, user_id, claim, subject, tokens, sources, entities, "
                "confidence, times_seen, episode_id, first_seen, last_seen, valid_to, "
                "superseded_by, expires_at, vector, vector_model) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?,?)",
                (
                    fact_id, episode.user_id, claim, subject,
                    json.dumps(list(tokens)), json.dumps(list(episode.sources)),
                    json.dumps(entities), confidence, 1, episode.id,
                    now, now, now + self.fact_ttl,
                    pack(vector) if vector else None, vector_model,
                ),
            )

            replaced = None
            if best is not None and best_similarity >= SUPERSEDE_SIMILARITY:
                # Same claim, different number: keep the history, recall only
                # the current value (Graphiti-style edge invalidation).
                conn.execute(
                    "UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?",
                    (now, fact_id, best["id"]),
                )
                replaced = best["id"]
            return FactOutcome("superseded" if replaced else "added", fact_id, replaced)

    def _update_profile(self, user_id: str, terms: Iterable[str], now: float) -> None:
        # Lowercase keys so "EU" and "eu" are one topic, not two.
        unique = list(dict.fromkeys(term.lower() for term in terms if term))
        if not unique:
            return
        with self._connect() as conn:
            for term in unique:
                conn.execute(
                    "INSERT INTO profile (user_id, term, weight, updated_at) VALUES (?,?,1.0,?) "
                    "ON CONFLICT(user_id, term) DO UPDATE SET "
                    "weight = weight * ? + 1, updated_at = excluded.updated_at",
                    (user_id, term, now, PROFILE_DECAY),
                )
            conn.execute(
                "DELETE FROM profile WHERE user_id = ? AND term NOT IN ("
                "SELECT term FROM profile WHERE user_id = ? ORDER BY weight DESC LIMIT ?)",
                (user_id, user_id, PROFILE_TERMS),
            )

    # ----------------------------------------------------------------- recall

    def recall(
        self,
        query: str,
        *,
        user_id: str = "default",
        session_id: str | None = None,
        k: int = 5,
    ) -> MemoryContext:
        """What memory knows that is relevant to ``query``, split by layer."""
        query = (query or "").strip()
        if not query or k <= 0:
            return MemoryContext()

        user = user_id or "default"
        now = self.clock()
        query_vector = self._query_vector(query)
        episode_hits = self._rank_episodes(
            self._episodes(user, now), query, now, session_id, query_vector
        )
        fact_hits = self._rank_facts(self._facts(user, now), query, now, query_vector)

        session_hits = [hit for hit in episode_hits if hit.layer == "session"]
        episodes = [hit for hit in episode_hits if hit.layer == "episode"][:k]

        # The working set is relevance-gated: only this session's runs that
        # actually connect to the new query. The most recent run is kept in
        # view only when nothing else in memory connects at all - a follow-up
        # like "and its latency?" that names nothing - so switching to a new
        # topic does not drag the previous one along.
        working_set = [hit for hit in session_hits if hit.matched][:SESSION_HITS]
        if not working_set and session_hits and not episodes and not fact_hits:
            working_set = [min(session_hits, key=lambda hit: hit.age_days)]

        # A follow-up that names nothing still gets the working set's facts,
        # so "and its latency?" is answered by what that run established.
        if not fact_hits and working_set:
            fact_hits = self._facts_for_episodes(
                [hit.id for hit in working_set], user, now, limit=2
            )

        return MemoryContext(
            session=working_set,
            episodes=episodes,
            facts=fact_hits[:k],
            profile=self.preferences(user, top=6),
        )

    # ------------------------------------------------------------- embeddings

    @property
    def _embedder_name(self) -> str:
        return self.embedder.name if self.embedder is not None else ""

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]] | None:
        """Batch-embed ``texts``, or ``None`` when unavailable.

        A failure (no key, quota, network) disables the route for the rest of
        the process so a broken embedder costs one attempt, not one per call.
        """
        if not texts or self.embedder is None or self._embedding_disabled:
            return None
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:
            self._embedding_disabled = True
            logger.warning("Memory embeddings unavailable (%s); using BM25", exc)
            return None
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            return None
        return [tuple(float(value) for value in vector) for vector in vectors]

    def _query_vector(self, query: str) -> tuple[float, ...]:
        vectors = self._embed([query])
        return vectors[0] if vectors else ()

    def backfill_vectors(self, *, user_id: str | None = None, limit: int = 200) -> int:
        """Embed records written before embeddings were enabled; returns count."""
        if self.embedder is None or limit <= 0:
            return 0

        name = self._embedder_name
        scope = "AND user_id = ? " if user_id else ""
        updated = 0

        with self._connect() as conn:
            episode_params = (name, user_id, limit) if user_id else (name, limit)
            episodes = conn.execute(
                "SELECT id, query, summary FROM episodes "
                "WHERE (vector IS NULL OR vector_model != ?) "
                + scope
                + "LIMIT ?",
                episode_params,
            ).fetchall()
            if episodes:
                vectors = self._embed(
                    [f"{row['query']}\n{row['summary']}" for row in episodes]
                )
                if vectors:
                    for row, vector in zip(episodes, vectors):
                        conn.execute(
                            "UPDATE episodes SET vector = ?, vector_model = ? WHERE id = ?",
                            (pack(vector), name, row["id"]),
                        )
                    updated += len(episodes)

            fact_params = (name, user_id, limit) if user_id else (name, limit)
            facts = conn.execute(
                "SELECT id, subject, claim FROM facts WHERE valid_to IS NULL "
                "AND (vector IS NULL OR vector_model != ?) "
                + scope
                + "LIMIT ?",
                fact_params,
            ).fetchall()
            if facts:
                vectors = self._embed(
                    [f"{row['subject']}\n{row['claim']}" for row in facts]
                )
                if vectors:
                    for row, vector in zip(facts, vectors):
                        conn.execute(
                            "UPDATE facts SET vector = ?, vector_model = ? WHERE id = ?",
                            (pack(vector), name, row["id"]),
                        )
                    updated += len(facts)

        return updated

    # ---------------------------------------------------------------- ranking

    def _hybrid_scores(
        self,
        query: str,
        docs: list[str],
        items: list,
        query_vector: tuple[float, ...],
    ) -> tuple[list[float], list[bool]]:
        """RRF-fused relevance per item, and whether either route matched."""
        lexical = bm25_scores(query, docs)
        name = self._embedder_name
        semantic: list[float | None] = [
            cosine(query_vector, item.vector)
            if query_vector and item.vector and item.vector_model == name
            else None
            for item in items
        ]

        # A route only ranks what it actually retrieved. Including zero-scored
        # records would hand every candidate a rank point and make relevance
        # meaningless (two unrelated facts could both normalise to 1.0).
        lexical_order = [
            index
            for index in sorted(range(len(docs)), key=lambda index: (-lexical[index], index))
            if lexical[index] > 0
        ]
        semantic_order = [
            index
            for index in sorted(
                (index for index, similarity in enumerate(semantic) if similarity is not None),
                key=lambda index: (-semantic[index], index),
            )
            if semantic[index] >= self.vector_min_cos
        ]
        fused = fuse_ranks([lexical_order, semantic_order])
        best = max(fused.values(), default=0.0)
        relevance = [
            (fused.get(index, 0.0) / best) if best > 0 else 0.0
            for index in range(len(docs))
        ]
        matched = [
            lexical[index] > 0
            or (semantic[index] is not None and semantic[index] >= self.vector_min_cos)
            for index in range(len(docs))
        ]
        return relevance, matched

    def _recency(self, hit: MemoryHit) -> float:
        half_life = RECENCY_HALF_LIFE_DAYS["fact" if hit.layer == "fact" else "episode"]
        return math.exp(-max(0.0, hit.age_days) / half_life)

    def _base_score(self, hit: MemoryHit) -> float:
        session = SESSION_BONUS if hit.layer == "session" else 0.0
        if hit.layer == "fact":
            return (
                0.55 * hit.relevance
                + 0.20 * self._recency(hit)
                + 0.25 * (hit.confidence or 0.0)
                + 0.15 * hit.entity
                + session
            )
        return (
            0.55 * hit.relevance
            + 0.25 * self._recency(hit)
            + 0.20 * hit.quality
            + session
        )

    @staticmethod
    def _hit_text(hit: MemoryHit) -> str:
        return f"{hit.query}\n{hit.text}" if hit.query else hit.text

    def _rerank(self, query: str, hits: list[MemoryHit]) -> list[MemoryHit]:
        """Order candidates by fused score, then by a cross-encoder when enabled."""
        scored = sorted(
            (replace(hit, score=self._base_score(hit)) for hit in hits),
            key=lambda hit: hit.score,
            reverse=True,
        )
        if self.reranker is None or len(scored) < 2:
            return scored

        candidates = scored[: self.rerank_candidates]
        rest = scored[self.rerank_candidates :]
        try:
            raw = self.reranker.score(
                query, [self._hit_text(hit) for hit in candidates]
            )
        except Exception as exc:
            logger.warning("Memory rerank unavailable (%s); using fused scores", exc)
            self.reranker = None
            return scored
        if len(raw) != len(candidates):
            return scored

        reranked = [
            replace(
                hit,
                score=(
                    0.70 * sigmoid(value)
                    + 0.15 * hit.relevance
                    + 0.10 * self._recency(hit)
                    + 0.05 * (hit.confidence if hit.layer == "fact" else hit.quality)
                    + (SESSION_BONUS if hit.layer == "session" else 0.0)
                ),
            )
            for hit, value in zip(candidates, raw)
        ]
        reranked.extend(rest)
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        return reranked

    def _rank_episodes(
        self,
        episodes: list[Episode],
        query: str,
        now: float,
        session_id: str | None,
        query_vector: tuple[float, ...] = (),
    ) -> list[MemoryHit]:
        if not episodes:
            return []
        docs = [f"{episode.query}\n{episode.summary}" for episode in episodes]
        relevance, matched = self._hybrid_scores(query, docs, episodes, query_vector)

        hits: list[MemoryHit] = []
        for episode, rel, ok in zip(episodes, relevance, matched):
            in_session = bool(session_id) and episode.session_id == session_id
            if not ok and not in_session:
                continue
            age_days = max(0.0, (now - episode.created_at) / 86_400)
            layer = "session" if in_session and age_days * 24 <= self.session_hours else "episode"
            hits.append(
                MemoryHit(
                    layer=layer,
                    id=episode.id,
                    text=episode.summary,
                    query=episode.query,
                    score=0.0,
                    age_days=age_days,
                    sources=episode.sources,
                    relevance=rel,
                    quality=min(1.0, len(episode.sources) / 8),
                    matched=ok,
                )
            )

        hits = self._rerank(query, hits)
        # Re-running the same question should not fill the budget with copies.
        picked: list[MemoryHit] = []
        seen: set[str] = set()
        for hit in hits:
            key = " ".join(hit.query.lower().split())
            if key in seen:
                continue
            seen.add(key)
            picked.append(hit)
        return picked

    def _rank_facts(
        self,
        facts: list[Fact],
        query: str,
        now: float,
        query_vector: tuple[float, ...] = (),
    ) -> list[MemoryHit]:
        if not facts:
            return []
        docs = [f"{fact.subject}\n{fact.claim}" for fact in facts]
        relevance, matched = self._hybrid_scores(query, docs, facts, query_vector)
        query_tokens = set(tokenize(query))

        hits: list[MemoryHit] = []
        for fact, rel, ok in zip(facts, relevance, matched):
            entity_overlap = len(query_tokens.intersection(tokenize(" ".join(fact.entities))))
            if not ok and not entity_overlap:
                continue
            hits.append(
                MemoryHit(
                    layer="fact",
                    id=fact.id,
                    text=fact.claim,
                    subject=fact.subject,
                    query="",
                    score=0.0,
                    age_days=max(0.0, (now - fact.last_seen) / 86_400),
                    confidence=fact.confidence,
                    times_seen=fact.times_seen,
                    sources=fact.sources,
                    relevance=rel,
                    quality=fact.confidence,
                    entity=min(1.0, entity_overlap / 2),
                    matched=ok or bool(entity_overlap),
                )
            )

        hits = self._rerank(query, hits)
        # One subject should not take the whole fact budget.
        picked: list[MemoryHit] = []
        per_subject: Counter[str] = Counter()
        for hit in hits:
            subject = hit.subject.lower()
            if per_subject[subject] >= 2:
                continue
            per_subject[subject] += 1
            picked.append(hit)
        return picked

    def preferences(self, user_id: str = "default", top: int = 6) -> list[str]:
        """Recurring topics, most persistent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT term FROM profile WHERE user_id = ? ORDER BY weight DESC LIMIT ?",
                (user_id or "default", max(0, int(top))),
            ).fetchall()
        return [row["term"] for row in rows]

    def _facts_for_episodes(
        self, episode_ids: list[str], user_id: str, now: float, limit: int = 2
    ) -> list[MemoryHit]:
        """Facts first reported by specific episodes, for follow-up continuity."""
        if not episode_ids:
            return []
        placeholders = ",".join("?" * len(episode_ids))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE user_id = ? AND valid_to IS NULL "
                f"AND expires_at > ? AND episode_id IN ({placeholders})",
                (user_id, now, *episode_ids),
            ).fetchall()
        facts = sorted(
            (_to_fact(row) for row in rows),
            key=lambda fact: fact.confidence,
            reverse=True,
        )[:limit]
        return [
            MemoryHit(
                layer="fact",
                id=fact.id,
                text=fact.claim,
                subject=fact.subject,
                score=fact.confidence,
                age_days=max(0.0, (now - fact.last_seen) / 86_400),
                confidence=fact.confidence,
                times_seen=fact.times_seen,
                sources=fact.sources,
                quality=fact.confidence,
                matched=True,
            )
            for fact in facts
        ]

    # --------------------------------------------------------------- lifecycle

    def forget_expired(self) -> int:
        """Delete expired rows; returns how many were removed."""
        now = self.clock()
        with self._connect() as conn:
            removed = conn.execute(
                "DELETE FROM episodes WHERE expires_at <= ?", (now,)
            ).rowcount
            removed += conn.execute(
                "DELETE FROM facts WHERE expires_at <= ?", (now,)
            ).rowcount
        return removed or 0

    def forget(self, *, user_id: str | None = None) -> int:
        """Delete all memory (for one user when ``user_id`` is given)."""
        with self._connect() as conn:
            if user_id is None:
                counts = [
                    conn.execute("DELETE FROM episodes").rowcount,
                    conn.execute("DELETE FROM facts").rowcount,
                    conn.execute("DELETE FROM profile").rowcount,
                ]
            else:
                counts = [
                    conn.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,)).rowcount,
                    conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,)).rowcount,
                    conn.execute("DELETE FROM profile WHERE user_id = ?", (user_id,)).rowcount,
                ]
        return sum(count or 0 for count in counts)

    def _prune_episodes(self, user_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM episodes WHERE id IN ("
                "SELECT id FROM episodes WHERE user_id = ? ORDER BY created_at DESC "
                "LIMIT -1 OFFSET ?)",
                (user_id, self.max_episodes),
            )
        return cursor.rowcount or 0

    def _prune_facts(self, user_id: str) -> int:
        # Superseded rows go first, then the least recently seen active ones.
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM facts WHERE id IN ("
                "SELECT id FROM facts WHERE user_id = ? "
                "ORDER BY (valid_to IS NULL), last_seen ASC LIMIT -1 OFFSET ?)",
                (user_id, self.max_facts),
            )
        return cursor.rowcount or 0

    def stats(self, user_id: str | None = None) -> dict[str, int]:
        """Row counts per layer, for observability and tests."""
        scope = " WHERE user_id = ?" if user_id else ""
        params: tuple = (user_id,) if user_id else ()

        with self._connect() as conn:
            def scalar(sql: str) -> int:
                return int(conn.execute(sql, params).fetchone()[0])

            return {
                "episodes": scalar(f"SELECT COUNT(*) FROM episodes{scope}"),
                "facts": scalar(
                    f"SELECT COUNT(*) FROM facts{scope}"
                    + (" AND" if scope else " WHERE")
                    + " valid_to IS NULL"
                ),
                "superseded": scalar(
                    f"SELECT COUNT(*) FROM facts{scope}"
                    + (" AND" if scope else " WHERE")
                    + " valid_to IS NOT NULL"
                ),
                "profile_terms": scalar(f"SELECT COUNT(*) FROM profile{scope}"),
            }

    # ----------------------------------------------------------------- loading

    def _episodes(self, user_id: str, now: float) -> list[Episode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE user_id = ? AND expires_at > ?",
                (user_id, now),
            ).fetchall()
        return [_to_episode(row) for row in rows]

    def _facts(self, user_id: str, now: float) -> list[Fact]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE user_id = ? AND valid_to IS NULL AND expires_at > ?",
                (user_id, now),
            ).fetchall()
        return [_to_fact(row) for row in rows]


#: v1 name, kept so existing imports keep working.
LongTermMemory = ResearchMemory


def _to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        query=row["query"],
        summary=row["summary"],
        sources=tuple(json.loads(row["sources"])),
        keywords=tuple(json.loads(row["keywords"])),
        entities=tuple(json.loads(row["entities"])),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        vector=unpack(row["vector"]),
        vector_model=row["vector_model"] or "",
    )


def _to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=row["id"],
        user_id=row["user_id"],
        claim=row["claim"],
        subject=row["subject"],
        tokens=tuple(json.loads(row["tokens"])),
        sources=tuple(json.loads(row["sources"])),
        entities=tuple(json.loads(row["entities"])),
        confidence=row["confidence"],
        times_seen=row["times_seen"],
        episode_id=row["episode_id"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        expires_at=row["expires_at"],
        vector=unpack(row["vector"]),
        vector_model=row["vector_model"] or "",
    )


def _subject_for(claim: str, entities: list[str], episode: Episode) -> str:
    """The entity a claim is about: first one mentioned, else the run's topic."""
    lowered = claim.lower()
    for entity in entities:
        if entity.lower() in lowered:
            return entity
    if entities:
        return entities[0]
    return episode.keywords[0] if episode.keywords else episode.query[:60]


def _conflicts(previous: frozenset[str], current: frozenset[str]) -> bool:
    """Two number sets disagree when both are non-empty and differ."""
    return bool(previous) and bool(current) and previous != current


# --------------------------------------------------------------------------- rendering


def _age_label(age_days: float) -> str:
    if age_days < 1:
        return "today"
    if age_days < 45:
        return f"{int(age_days)}d"
    return f"{int(age_days / 30)}mo"


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if limit <= 1 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ,;，；:：") + "…"


def _sources_label(count: int) -> str:
    return f"{count} source" if count == 1 else f"{count} sources"


def format_memory_block(context: MemoryContext | None, max_chars: int = 2_000) -> str:
    """Render recalled memory as a prompt block, or ``""`` when there is none."""
    if context is None or context.is_empty():
        return ""

    header = [
        "## Prior research memory",
        "Background from this session and earlier runs. Reuse it, verify anything",
        "time-sensitive, and do not cite it as a source.",
    ]
    lines = list(header)
    budget = max_chars - sum(len(line) + 1 for line in lines)
    if budget <= 0:
        return ""

    def section(title: str, items: list[str], share: float) -> None:
        nonlocal budget
        if not items or budget <= 0:
            return
        rendered = ["", f"### {title}"]
        cost = sum(len(line) + 1 for line in rendered)
        available = min(budget, int(max_chars * share))
        if available - cost <= 0:
            return
        for item in items:
            line = f"- {_clip(item, 240)}"
            if cost + len(line) + 1 > available:
                break
            rendered.append(line)
            cost += len(line) + 1
        if len(rendered) > 2:
            lines.extend(rendered)
            budget -= cost

    section(
        "This session",
        [f"{hit.query}: {hit.text}" for hit in context.session],
        share=0.25,
    )
    section(
        "Established facts",
        [
            f"{hit.text} [{_age_label(hit.age_days)}, conf {hit.confidence:.2f}"
            + (f", {hit.times_seen}×" if hit.times_seen > 1 else "")
            + "]"
            for hit in context.facts
        ],
        share=0.40,
    )
    section(
        "Earlier research",
        [
            f"({_age_label(hit.age_days)}, {_sources_label(len(hit.sources))}) {hit.query}: {hit.text}"
            for hit in context.episodes
        ],
        share=0.40,
    )
    if context.profile and budget > 0:
        profile = "Focus: " + ", ".join(context.profile[:6])
        if len(profile) + 1 <= budget:
            lines.extend(["", _clip(profile, 200)])

    return "\n".join(lines)


def memory_from_config(cfg) -> ResearchMemory | None:
    """Build the store from config, or ``None`` when the feature is off."""
    if not bool(getattr(cfg, "long_term_memory", False)):
        return None

    path = getattr(cfg, "memory_db_path", None) or os.path.join(
        os.path.expanduser("~"), ".gpt_researcher", "memory.sqlite3"
    )
    return ResearchMemory(
        path,
        episode_ttl_days=float(getattr(cfg, "memory_ttl_days", 30) or 30),
        fact_ttl_days=float(getattr(cfg, "memory_fact_ttl_days", 365) or 365),
        session_hours=float(getattr(cfg, "memory_session_ttl_hours", 24) or 24),
        max_episodes=int(getattr(cfg, "memory_max_records", 500) or 500),
        max_facts=int(getattr(cfg, "memory_max_facts", 1_000) or 1_000),
        facts_per_run=int(getattr(cfg, "memory_facts_per_run", 6) or 6),
        embedder=build_embedder(cfg),
        reranker=build_reranker(cfg),
        rerank_candidates=int(getattr(cfg, "memory_rerank_candidates", 24) or 24),
        vector_min_cos=float(getattr(cfg, "memory_vector_min_cos", 0.3) or 0.3),
    )
