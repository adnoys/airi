"""Tests for keyword scoring, text dedup helpers, and the LLM reranker."""

from __future__ import annotations

import pytest

from airi_memory.llm import LLMClient
from airi_memory.scoring import dedup_key, keyword_scores, normalize_text


class TestKeywordScores:
    def test_exact_word_match_scores_highest(self) -> None:
        entries = [{'content': 'User likes matcha'}, {'content': 'User owns a cat'}]
        scores = keyword_scores('matcha', entries)
        assert scores[0] > scores[1]

    def test_extra_entry_terms_lower_the_overlap_score(self) -> None:
        entries = [{'content': 'matcha'}, {'content': 'matcha latte every morning'}]
        exact, longer = keyword_scores('matcha', entries)
        assert exact == pytest.approx(1.0)
        assert longer < exact

    def test_chinese_bigram_overlap(self) -> None:
        entries = [{'content': '用户最喜欢吃红烧肉'}, {'content': '用户养了一只猫'}]
        scores = keyword_scores('红烧肉', entries)
        assert scores[0] > 0
        assert scores[1] == 0

    def test_paraphrase_without_overlap_scores_zero(self) -> None:
        # This is why the LLM rerank stage exists: keyword recall is the
        # coarse stage only.
        entries = [{'content': '用户最喜欢吃红烧肉'}]
        assert keyword_scores('我爱吃的菜', entries) == [0.0]

    def test_empty_entries_return_no_scores(self) -> None:
        assert keyword_scores('anything', []) == []


class TestNormalizeText:
    def test_folds_whitespace_and_case(self) -> None:
        assert normalize_text('  User  likes  Matcha ') == 'userlikesmatcha'

    def test_different_texts_differ(self) -> None:
        assert normalize_text('a b c') != normalize_text('a b d')


class TestDedupKey:
    def test_ignores_the_parenthetical_quote(self) -> None:
        assert dedup_key('用户喜欢猫（用户原话：我养了一只猫）') == dedup_key('用户喜欢猫')

    def test_folds_whitespace_and_case_like_normalize(self) -> None:
        assert dedup_key('User  likes matcha') == dedup_key('user likes matcha')

    def test_different_facts_differ(self) -> None:
        assert dedup_key('用户喜欢猫（原话：我养猫）') != dedup_key('用户喜欢狗（原话：我养狗）')

    def test_content_without_a_fact_falls_back_to_the_whole_text(self) -> None:
        assert dedup_key('（只有引用）') == dedup_key('（只有引用）')
        assert dedup_key('（只有引用）') != dedup_key('（另一句引用）')


class FakeChatTransport:
    """Feeds a canned JSON body to the real HTTP path via httpx transport."""

    def __init__(self, content: str) -> None:
        import httpx

        self.handler = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={
                'choices': [{'message': {'content': content}}],
            }),
        )


def make_client(content: str) -> LLMClient:
    import httpx

    client = LLMClient('https://llm.example.com/v1', 'key', 'fake-model')
    client._client = httpx.AsyncClient(transport=FakeChatTransport(content).handler)
    return client


class TestLLMRerank:
    @pytest.mark.asyncio
    async def test_parses_numbered_ranking(self) -> None:
        client = make_client('[2, 0]')
        picked = await client.rerank('query', ['a', 'b', 'c'], top_k=2)
        assert picked == [2, 0]

    @pytest.mark.asyncio
    async def test_tolerates_fences_and_prose(self) -> None:
        client = make_client('Here you go:\n```json\n[1]\n```\nDone.')
        picked = await client.rerank('query', ['a', 'b'], top_k=3)
        assert picked == [1]

    @pytest.mark.asyncio
    async def test_drops_out_of_range_and_duplicate_indices(self) -> None:
        client = make_client('[5, 0, 0, 1]')
        picked = await client.rerank('query', ['a', 'b'], top_k=5)
        assert picked == [0, 1]

    @pytest.mark.asyncio
    async def test_caps_results_at_top_k(self) -> None:
        client = make_client('[0, 1, 2, 3]')
        picked = await client.rerank('query', ['a', 'b', 'c', 'd'], top_k=2)
        assert picked == [0, 1]

    @pytest.mark.asyncio
    async def test_garbled_reply_ranks_nothing(self) -> None:
        client = make_client('I cannot help with that.')
        picked = await client.rerank('query', ['a', 'b'], top_k=2)
        assert picked == []

    @pytest.mark.asyncio
    async def test_empty_message_raises(self) -> None:
        client = make_client('')
        with pytest.raises(Exception, match='empty message'):
            await client.rerank('query', ['a'], top_k=1)
