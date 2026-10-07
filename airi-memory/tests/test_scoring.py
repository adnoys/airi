"""Unit tests for the stateless scoring functions."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from airi_memory import scoring


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()


def make_entry(created_at: str, strength: float = 1.0, half_life_days: float = 7.0) -> dict:
    return {
        'id': 'entry',
        'content': 'content',
        'embedding': [1.0, 0.0],
        'strength': strength,
        'half_life_days': half_life_days,
        'recall_count': 0,
        'created_at': created_at,
        'last_recalled_at': None,
        'source_session_id': None,
    }


class TestDecay:
    def test_strength_halves_after_one_half_life(self) -> None:
        assert scoring.decayed_strength(1.0, 7.0, 7.0) == pytest.approx(0.5)

    def test_strength_quarters_after_two_half_lives(self) -> None:
        assert scoring.decayed_strength(1.0, 7.0, 14.0) == pytest.approx(0.25)

    def test_fresh_memory_keeps_full_strength(self) -> None:
        assert scoring.decayed_strength(1.0, 7.0, 0.0) == pytest.approx(1.0)

    def test_zero_half_life_decays_to_zero(self) -> None:
        assert scoring.decayed_strength(1.0, 0.0, 1.0) == 0.0


class TestAge:
    def test_age_from_created_at(self) -> None:
        now = 1_700_000_000.0
        created = now - 3 * scoring.SECONDS_PER_DAY
        assert scoring.age_in_days(iso(created), now) == pytest.approx(3.0)

    def test_future_created_at_clamps_to_zero(self) -> None:
        now = 1_700_000_000.0
        created = now + 100
        assert scoring.age_in_days(iso(created), now) == 0.0


class TestKind:
    def test_default_half_life_is_short(self) -> None:
        assert scoring.memory_kind(7.0) == 'short'

    def test_at_threshold_is_long(self) -> None:
        assert scoring.memory_kind(14.0) == 'long'

    def test_above_threshold_is_long(self) -> None:
        assert scoring.memory_kind(365.0) == 'long'


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert scoring.cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert scoring.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert scoring.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_scores_zero(self) -> None:
        assert scoring.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match='length mismatch'):
            scoring.cosine_similarity([1.0], [1.0, 0.0])


class TestReinforce:
    def test_recall_count_increments(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now))
        reinforced = scoring.reinforce(entry, now)
        assert reinforced['recall_count'] == 1
        assert reinforced['last_recalled_at'] is not None

    def test_half_life_grows_and_strength_rises(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now), strength=1.0, half_life_days=7.0)
        reinforced = scoring.reinforce(entry, now)
        assert reinforced['half_life_days'] == pytest.approx(7.0 * scoring.REINFORCE_HALF_LIFE_FACTOR)
        assert reinforced['strength'] == pytest.approx(1.0 + scoring.REINFORCE_STRENGTH_DELTA)

    def test_original_entry_untouched(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now))
        scoring.reinforce(entry, now)
        assert entry['recall_count'] == 0
        assert entry['half_life_days'] == 7.0

    def test_half_life_caps_at_max(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now), half_life_days=scoring.MAX_HALF_LIFE_DAYS)
        reinforced = scoring.reinforce(entry, now)
        assert reinforced['half_life_days'] == scoring.MAX_HALF_LIFE_DAYS


class TestRankEntries:
    def test_entry_without_embedding_never_matches(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now))
        entry['embedding'] = None
        assert scoring.rank_entries([1.0, 0.0], [entry], now) == []

    def test_similarity_outranks_recency(self) -> None:
        now = 1_700_000_000.0
        near = make_entry(iso(now - 90 * scoring.SECONDS_PER_DAY), half_life_days=1.0)
        near['id'] = 'near'
        near['embedding'] = [1.0, 0.0]
        far_but_fresh = make_entry(iso(now), half_life_days=1.0)
        far_but_fresh['id'] = 'far_but_fresh'
        far_but_fresh['embedding'] = [0.0, 1.0]
        ranked = scoring.rank_entries([1.0, 0.0], [far_but_fresh, near], now)
        assert ranked[0]['id'] == 'near'
        assert ranked[0]['similarity'] == pytest.approx(1.0)

    def test_equal_similarity_prefers_fresher_memory(self) -> None:
        now = 1_700_000_000.0
        fresh = make_entry(iso(now))
        old = make_entry(iso(now - 90 * scoring.SECONDS_PER_DAY))
        fresh['id'] = 'fresh'
        old['id'] = 'old'
        ranked = scoring.rank_entries([1.0, 0.0], [old, fresh], now)
        assert ranked[0]['id'] == 'fresh'
        assert ranked[0]['time_relevance'] > ranked[1]['time_relevance']

    def test_score_follows_devlog_weights(self) -> None:
        now = 1_700_000_000.0
        entry = make_entry(iso(now))
        entry['embedding'] = [1.0, 0.0]
        ranked = scoring.rank_entries([1.0, 0.0], [entry], now)
        expected = scoring.SIMILARITY_WEIGHT * 1.0 + scoring.TIME_RELEVANCE_WEIGHT * 1.0
        assert ranked[0]['score'] == pytest.approx(expected)
        assert not math.isnan(ranked[0]['time_relevance'])
