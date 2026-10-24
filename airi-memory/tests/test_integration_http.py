"""Integration tests over real HTTP sockets.

``TestClient`` bypasses CORS and the network stack, so the browser-facing
contract is proven here against uvicorn on an ephemeral port with a
throwaway data directory. The user's real service on 6430 is never touched.
"""

from __future__ import annotations

import json
import threading
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from airi_memory.config import EmbeddingConfig, LLMConfig, ServiceConfig
from airi_memory.server import create_app
from airi_memory.store import MemoryStore


class RecordingReranker:
    """Returns every candidate, the worst case, and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(self, query: str, candidates: list[str], top_k: int) -> list[int]:
        self.calls.append((query, list(candidates), top_k))
        return list(range(len(candidates)))

    async def close(self) -> None:
        pass


class FixedEmbedder:
    """Deterministic one-hot vector from a CRC slot of the text."""

    def vector_for(self, text: str) -> list[float]:
        vector = [0.0] * 8
        vector[zlib.crc32(text.encode('utf-8')) % 8] = 1.0
        return vector

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.vector_for(text) for text in texts]

    async def embed(self, text: str) -> list[float]:
        return self.vector_for(text)

    async def close(self) -> None:
        pass


def start_server(app) -> tuple[uvicorn.Server, threading.Thread, int]:
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError('uvicorn did not start')
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, port


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def llm_service(tmp_path: Path):
    reranker = RecordingReranker()
    config = ServiceConfig(
        data_dir=tmp_path,
        retrieval='llm',
        llm=LLMConfig(base_url='https://api.deepseek.com', api_key='llm-secret', model='deepseek-chat'),
    )
    server, thread, port = start_server(create_app(config, MemoryStore(tmp_path), llm_factory=lambda _c: reranker))
    yield SimpleNamespace(port=port, reranker=reranker, data_dir=tmp_path)
    stop_server(server, thread)


def post_memory(client: httpx.Client, content: str) -> dict:
    response = client.post('/v1/memories', json={'content': content})
    assert response.status_code == 200
    return response.json()


class TestReadOnlySearch:
    """Problem 7: search must never write, over the real protocol too."""

    def test_repeated_search_leaves_storage_untouched(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            post_memory(client, '用户喜欢猫（用户原话：我养了一只猫）')
            store_file = llm_service.data_dir / 'memories.json'
            before = store_file.read_bytes()
            for _ in range(3):
                body = client.post('/v1/search', json={'query': '猫', 'top_k': 6}).json()
                assert body['results'][0]['recall_count'] == 0
                assert body['results'][0]['half_life_days'] == 7.0
                assert body['results'][0]['last_recalled_at'] is None
            assert store_file.read_bytes() == before


class TestNoReadTimeRevision:
    """Problem 2: the exact memory the old revision layer deleted survives."""

    def test_paraphrased_memory_is_returned_intact(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            post_memory(client, '用户昨天问过病娇是什么。')
            body = client.post('/v1/search', json={'query': '在乎', 'top_k': 6}).json()
        contents = [entry['content'] for entry in body['results']]
        assert '用户昨天问过病娇是什么。' in contents


class TestNamedDayGone:
    """Problem 1: a query naming a day ranks like any other query."""

    def test_day_query_goes_through_the_reranker(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            post_memory(client, '用户想要鲸鱼的图片')
            post_memory(client, '用户说想听我爱你')
            post_memory(client, '用户喜欢抹茶')
            for query in ('你今天过得怎么样', '我前天说了什么'):
                body = client.post('/v1/search', json={'query': query, 'top_k': 2}).json()
                # The permissive reranker returns every candidate; the
                # service must still cap the answer at top_k.
                assert len(body['results']) == 2

    def test_every_candidate_carries_its_local_date(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            post_memory(client, '用户喜欢抹茶')
            client.post('/v1/search', json={'query': '我前天说了什么', 'top_k': 6})
        for line in llm_service.reranker.calls[-1][1]:
            assert line.startswith('[20')
            assert '] ' in line


class TestRestatementReinforces:
    """Problem 8: one fact restated with a new quote reinforces, not duplicates."""

    def test_same_fact_different_quote_reinforces(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            first = post_memory(client, '用户喜欢猫（用户原话：我养了一只猫）')['entry']
            restated = post_memory(client, '用户喜欢猫（用户原话：家里有那只猫）')
            other = post_memory(client, '用户喜欢狗（用户原话：狗最忠诚）')
        assert restated['deduplicated'] is True
        assert restated['entry']['id'] == first['id']
        assert restated['entry']['recall_count'] == 1
        assert restated['entry']['half_life_days'] == 8.75
        assert other['deduplicated'] is False
        assert other['entry']['id'] != first['id']


class TestConfigKeySafety:
    """Problem 3: keys never echo, and an empty key keeps the stored one."""

    def test_get_config_returns_no_key(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            body = client.get('/v1/config').json()
        assert body['llm']['api_key'] == ''
        assert body['llm']['model'] == 'deepseek-chat'

    def test_empty_key_keeps_the_stored_key_only_for_the_same_endpoint(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            client.put('/v1/config', json={
                'retrieval': 'llm',
                'llm': {'base_url': 'https://api.deepseek.com', 'api_key': 'fresh-key', 'model': 'deepseek-chat'},
            })
            client.put('/v1/config', json={
                'retrieval': 'llm',
                'llm': {'base_url': 'https://api.deepseek.com', 'api_key': '', 'model': 'deepseek-chat'},
            })
            kept = json.loads((llm_service.data_dir / 'config.json').read_text(encoding='utf-8'))
            client.put('/v1/config', json={
                'retrieval': 'llm',
                'llm': {'base_url': 'https://other.example.com', 'api_key': '', 'model': 'deepseek-chat'},
            })
            dropped = json.loads((llm_service.data_dir / 'config.json').read_text(encoding='utf-8'))
        assert kept['llm']['api_key'] == 'fresh-key'
        assert dropped['llm']['api_key'] == ''


class TestCorsLocalhostOnly:
    """Problem 3: a foreign web page cannot read the answers."""

    def test_foreign_origin_gets_no_cors_headers(self, llm_service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{llm_service.port}') as client:
            foreign = client.get('/v1/health', headers={'Origin': 'https://evil.example'})
            local = client.get('/v1/health', headers={'Origin': 'http://localhost:5173'})
            preflight_foreign = client.options('/v1/health', headers={
                'Origin': 'https://evil.example',
                'Access-Control-Request-Method': 'GET',
            })
            preflight_local = client.options('/v1/health', headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'GET',
            })
        assert 'access-control-allow-origin' not in foreign.headers
        assert local.headers['access-control-allow-origin'] == 'http://localhost:5173'
        assert 'access-control-allow-origin' not in preflight_foreign.headers
        assert preflight_local.headers['access-control-allow-origin'] == 'http://localhost:5173'


class TestBackfillAndOmittedClient:
    """Problems 10 and 13: batch backfill, and PUT keeps omitted clients."""

    @pytest.fixture()
    def service(self, tmp_path: Path):
        reranker = RecordingReranker()
        config = ServiceConfig(
            data_dir=tmp_path,
            retrieval='llm',
            llm=LLMConfig(base_url='https://api.deepseek.com', api_key='llm-secret', model='deepseek-chat'),
        )
        app = create_app(
            config,
            MemoryStore(tmp_path),
            embedder_factory=lambda _c: FixedEmbedder(),
            llm_factory=lambda _c: reranker,
        )
        server, thread, port = start_server(app)
        yield SimpleNamespace(port=port, data_dir=tmp_path)
        stop_server(server, thread)

    def test_mode_switch_backfills_in_batches_and_keeps_the_llm(self, service) -> None:
        with httpx.Client(base_url=f'http://127.0.0.1:{service.port}') as client:
            for content in ('用户喜欢抹茶', '用户养了一只猫', '用户住在京都'):
                post_memory(client, content)
            body = client.put('/v1/config', json={
                'retrieval': 'embedding',
                'embedding': {'base_url': 'https://embed.example.com/v1', 'api_key': 'embed-key', 'model': 'embed'},
            }).json()
            health = client.get('/v1/health').json()
            search = client.post('/v1/search', json={'query': '抹茶', 'top_k': 6})
        assert body['backfilled'] == 3
        assert body['embedding']['api_key'] == ''
        assert health['embedding_configured'] is True
        assert health['llm_configured'] is True
        assert search.status_code == 200
