"""Tests for the OpenAI-compatible embedding client."""

from __future__ import annotations

import json

import httpx
import pytest

from airi_memory.embedding import EmbeddingClient, EmbeddingError


def client_with(handler) -> EmbeddingClient:
    client = EmbeddingClient('https://embed.example/v1/', 'key', 'fake-model')
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def json_body(request: httpx.Request) -> dict:
    return json.loads(request.content)


class TestEmbeddingClient:
    @pytest.mark.asyncio
    async def test_empty_input_skips_the_request(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError('empty input must not call the API')

        vectors = await client_with(handler).embed_many([])

        assert vectors == []

    @pytest.mark.asyncio
    async def test_returns_vectors_in_input_order(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == '/v1/embeddings'
            assert request.headers['authorization'] == 'Bearer key'
            assert json_body(request) == {'model': 'fake-model', 'input': ['first', 'second']}
            return httpx.Response(200, json={
                'data': [
                    {'index': 1, 'embedding': [0.0, 1.0]},
                    {'index': 0, 'embedding': [1.0, 0.0]},
                ],
            })

        vectors = await client_with(handler).embed_many(['first', 'second'])

        assert vectors == [[1.0, 0.0], [0.0, 1.0]]

    @pytest.mark.asyncio
    async def test_embed_returns_the_single_vector(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [0.5]}]})

        vector = await client_with(handler).embed('one')

        assert vector == [0.5]

    @pytest.mark.asyncio
    async def test_http_error_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={'error': 'down'})

        with pytest.raises(EmbeddingError, match='request failed'):
            await client_with(handler).embed('one')

    @pytest.mark.asyncio
    async def test_wrong_vector_count_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0]}]})

        with pytest.raises(EmbeddingError, match='1 vectors for 2 inputs'):
            await client_with(handler).embed_many(['a', 'b'])

