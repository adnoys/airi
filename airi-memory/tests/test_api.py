"""API tests with an injected fake embedder, exercising the full protocol."""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from airi_memory.config import EmbeddingConfig, ServiceConfig
from airi_memory.server import create_app
from airi_memory.store import MemoryStore

VECTOR_DIMENSIONS = 8


class FakeEmbedder:
    """Deterministic embedder: one-hot vector from a CRC of the text.

    Texts that hash to the same slot get identical vectors, so tests can
    control similarity by choosing texts. ``close`` counts calls to let
    tests assert shutdown behavior if needed.
    """

    def __init__(self) -> None:
        self.close_calls = 0

    def vector_for(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_DIMENSIONS
        vector[zlib.crc32(text.encode('utf-8')) % VECTOR_DIMENSIONS] = 1.0
        return vector

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.vector_for(text) for text in texts]

    async def embed(self, text: str) -> list[float]:
        return self.vector_for(text)

    async def close(self) -> None:
        self.close_calls += 1


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def client(tmp_path: Path, fake_embedder: FakeEmbedder) -> TestClient:
    config = ServiceConfig(
        data_dir=tmp_path,
        embedding=EmbeddingConfig(base_url='https://embed.example.com/v1', model='fake-model'),
    )
    app = create_app(config, MemoryStore(tmp_path), embedder_factory=lambda _config: fake_embedder)
    with TestClient(app) as test_client:
        yield test_client


def add_memory(client: TestClient, content: str) -> dict:
    response = client.post('/v1/memories', json={'content': content})
    assert response.status_code == 200
    return response.json()


class TestHealth:
    def test_reports_ok_without_embedding(self, tmp_path: Path) -> None:
        app = create_app(ServiceConfig(data_dir=tmp_path), MemoryStore(tmp_path))
        with TestClient(app) as test_client:
            body = test_client.get('/v1/health').json()
        assert body['status'] == 'ok'
        assert body['embedding_configured'] is False

    def test_reports_embedding_configured(self, client: TestClient) -> None:
        body = client.get('/v1/health').json()
        assert body['embedding_configured'] is True


class TestConfig:
    def test_put_config_backfills_missing_embeddings(self, tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
        store = MemoryStore(tmp_path)
        store.add('Stored before config existed', None)
        app = create_app(ServiceConfig(data_dir=tmp_path), store, embedder_factory=lambda _c: fake_embedder)
        with TestClient(app) as test_client:
            body = test_client.put('/v1/config', json={
                'embedding': {'base_url': 'https://embed.example.com/v1', 'api_key': 'k', 'model': 'm'},
            }).json()
            assert body['backfilled'] == 1
            entries = test_client.get('/v1/memories').json()['entries']
            assert entries[0]['content'] == 'Stored before config existed'

    def test_get_config_roundtrip(self, client: TestClient) -> None:
        body = client.get('/v1/config').json()
        assert body['embedding']['model'] == 'fake-model'


class TestMemories:
    def test_create_returns_public_entry(self, client: TestClient) -> None:
        body = add_memory(client, 'User likes matcha')
        entry = body['entry']
        assert body['deduplicated'] is False
        assert entry['content'] == 'User likes matcha'
        assert 'embedding' not in entry
        assert entry['half_life_days'] == 7.0

    def test_identical_content_deduplicates_and_reinforces(self, client: TestClient) -> None:
        first = add_memory(client, 'User likes matcha')['entry']
        second = add_memory(client, 'User likes matcha')
        assert second['deduplicated'] is True
        assert second['entry']['id'] == first['id']
        assert second['entry']['recall_count'] == 1

    def test_create_without_embedding_config_fails(self, tmp_path: Path) -> None:
        app = create_app(ServiceConfig(data_dir=tmp_path), MemoryStore(tmp_path))
        with TestClient(app) as test_client:
            response = test_client.post('/v1/memories', json={'content': 'Anything'})
        assert response.status_code == 503
        assert 'Embedding is not configured' in response.json()['detail']

    def test_patch_reembeds(self, client: TestClient) -> None:
        entry = add_memory(client, 'Before patch')['entry']
        patched = client.patch(f"/v1/memories/{entry['id']}", json={'content': 'After patch'}).json()['entry']
        assert patched['content'] == 'After patch'

    def test_patch_unknown_returns_404(self, client: TestClient) -> None:
        response = client.patch('/v1/memories/missing', json={'content': 'X'})
        assert response.status_code == 404

    def test_delete_single(self, client: TestClient) -> None:
        entry = add_memory(client, 'Delete me')['entry']
        assert client.delete(f"/v1/memories/{entry['id']}").json()['deleted'] == entry['id']
        assert client.delete(f"/v1/memories/{entry['id']}").status_code == 404

    def test_clear_all(self, client: TestClient) -> None:
        add_memory(client, 'One')
        add_memory(client, 'Two')
        assert client.delete('/v1/memories').json()['deleted'] == 2
        assert client.get('/v1/memories').json()['entries'] == []


class TestSearch:
    def test_search_returns_scored_results(self, client: TestClient) -> None:
        add_memory(client, 'alpha')
        add_memory(client, 'beta')
        body = client.post('/v1/search', json={'query': 'alpha', 'top_k': 5}).json()
        assert len(body['results']) == 2
        best = body['results'][0]
        assert best['content'] == 'alpha'
        assert best['similarity'] == pytest.approx(1.0)
        assert best['score'] >= best['similarity']

    def test_search_reinforces_hits(self, client: TestClient) -> None:
        add_memory(client, 'alpha')
        first = client.post('/v1/search', json={'query': 'alpha'}).json()['results'][0]
        assert first['recall_count'] == 1
        second = client.post('/v1/search', json={'query': 'alpha'}).json()['results'][0]
        assert second['recall_count'] == 2

    def test_top_k_limits_results(self, client: TestClient) -> None:
        add_memory(client, 'alpha')
        add_memory(client, 'beta')
        add_memory(client, 'gamma')
        body = client.post('/v1/search', json={'query': 'alpha', 'top_k': 1}).json()
        assert len(body['results']) == 1

    def test_search_without_embedding_config_fails(self, tmp_path: Path) -> None:
        app = create_app(ServiceConfig(data_dir=tmp_path), MemoryStore(tmp_path))
        with TestClient(app) as test_client:
            response = test_client.post('/v1/search', json={'query': 'Anything'})
        assert response.status_code == 503
