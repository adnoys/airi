"""Stateless memory scoring.

Implements the decay and reinforcement design from the AIRI memory
devlogs (DevLog 2025.04.06 / 2025.04.14):

- Scores decay by a half-life, computed on demand from the current time,
  so no background job ever rewrites stored scores.
- A recall reinforces a memory: it gains strength and a longer half-life,
  which is how a short-term memory evolves into a long-term one.
- Recall ranking combines vector similarity (coarse) with time relevance
  (fine): ``1.2 * similarity + 0.2 * time_relevance``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

SECONDS_PER_DAY = 86400.0

INITIAL_STRENGTH = 1.0
MAX_STRENGTH = 10.0
INITIAL_HALF_LIFE_DAYS = 7.0
MAX_HALF_LIFE_DAYS = 365.0
LONG_TERM_HALF_LIFE_DAYS = 14.0

REINFORCE_HALF_LIFE_FACTOR = 1.25
REINFORCE_STRENGTH_DELTA = 0.1

SIMILARITY_WEIGHT = 1.2
TIME_RELEVANCE_WEIGHT = 0.2

DEDUP_SIMILARITY_THRESHOLD = 0.92

KIND_SHORT = 'short'
KIND_LONG = 'long'


def memory_kind(half_life_days: float) -> str:
    """Classify a memory by how slow it decays."""
    return KIND_LONG if half_life_days >= LONG_TERM_HALF_LIFE_DAYS else KIND_SHORT


def parse_timestamp(value: str) -> float:
    """Convert an ISO 8601 UTC timestamp to a Unix timestamp in seconds."""
    return datetime.fromisoformat(value).timestamp()


def age_in_days(created_at: str, now_seconds: float) -> float:
    """Age of a memory in days, never negative."""
    age_seconds = max(0.0, now_seconds - parse_timestamp(created_at))
    return age_seconds / SECONDS_PER_DAY


def decayed_strength(strength: float, half_life_days: float, age_days: float) -> float:
    """Strength as of now, halved once per half-life. Stateless on purpose."""
    if half_life_days <= 0:
        return 0.0
    return strength * 0.5 ** (age_days / half_life_days)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. Zero when either is zero."""
    if len(a) != len(b):
        raise ValueError(f'Vector length mismatch: {len(a)} vs {len(b)}')
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def reinforce(entry: dict, now_seconds: float) -> dict:
    """Return a copy of ``entry`` after one recall.

    Reinforcement grows strength and half-life, so recalled memories fade
    slower and eventually cross the long-term threshold.
    """
    next_entry = {**entry}
    next_entry['strength'] = min(
        MAX_STRENGTH,
        float(entry.get('strength', INITIAL_STRENGTH)) + REINFORCE_STRENGTH_DELTA,
    )
    next_entry['half_life_days'] = min(
        MAX_HALF_LIFE_DAYS,
        float(entry.get('half_life_days', INITIAL_HALF_LIFE_DAYS)) * REINFORCE_HALF_LIFE_FACTOR,
    )
    next_entry['recall_count'] = int(entry.get('recall_count', 0)) + 1
    next_entry['last_recalled_at'] = datetime.fromtimestamp(
        now_seconds, tz=timezone.utc,
    ).isoformat()
    return next_entry


def time_relevance(entry: dict, now_seconds: float, max_decayed: float) -> float:
    """Decayed strength of one entry, normalized against the best candidate."""
    if max_decayed <= 0:
        return 0.0
    age = age_in_days(entry['created_at'], now_seconds)
    decayed = decayed_strength(
        float(entry.get('strength', INITIAL_STRENGTH)),
        float(entry.get('half_life_days', INITIAL_HALF_LIFE_DAYS)),
        age,
    )
    return decayed / max_decayed


def rank_entries(
    query_embedding: list[float],
    entries: list[dict],
    now_seconds: float,
) -> list[dict]:
    """Rank entries for a query and return scored copies, best first.

    Each result is ``{**entry, 'score', 'similarity', 'time_relevance'}``.
    Entries without an embedding never match.
    """
    candidates = []
    for entry in entries:
        embedding = entry.get('embedding')
        if embedding:
            similarity = cosine_similarity(query_embedding, embedding)
            age = age_in_days(entry['created_at'], now_seconds)
            decayed = decayed_strength(
                float(entry.get('strength', INITIAL_STRENGTH)),
                float(entry.get('half_life_days', INITIAL_HALF_LIFE_DAYS)),
                age,
            )
            candidates.append((entry, similarity, decayed))

    max_decayed = max((decayed for _, _, decayed in candidates), default=0.0)

    results = []
    for entry, similarity, _ in candidates:
        relevance = time_relevance(entry, now_seconds, max_decayed)
        score = SIMILARITY_WEIGHT * similarity + TIME_RELEVANCE_WEIGHT * relevance
        results.append({**entry, 'score': score, 'similarity': similarity, 'time_relevance': relevance})

    results.sort(key=lambda item: item['score'], reverse=True)
    return results


def _text_bigrams(text: str) -> set[str]:
    """Character bigrams plus lowercase latin word tokens.

    Bigrams make keyword matching work for CJK text without a
    segmentation model; word tokens keep English matching exact.
    """
    tokens = set(re.findall(r'[a-z0-9]+', text.lower()))
    compact = re.sub(r'\s+', '', text)
    tokens |= {compact[i:i + 2] for i in range(len(compact) - 1)}
    return tokens


def keyword_scores(query: str, entries: list[dict]) -> list[float]:
    """Coarse keyword relevance of each entry against the query.

    Returns one overlap score in ``[0, 1]`` per entry, in input order.
    This is the cheap first stage when no embedding model is available;
    the chat model reranks the best candidates afterwards.
    """
    if not entries:
        return []
    query_terms = _text_bigrams(query)
    if not query_terms:
        return [0.0] * len(entries)
    scores = []
    for entry in entries:
        entry_terms = _text_bigrams(str(entry.get('content', '')))
        if not entry_terms:
            scores.append(0.0)
            continue
        overlap = query_terms & entry_terms
        scores.append(len(overlap) / max(1, len(query_terms & entry_terms | query_terms)))
    return scores


def normalize_text(content: str) -> str:
    """Fold a memory's content for exact-text duplicate checks."""
    return ''.join(content.split()).lower()
