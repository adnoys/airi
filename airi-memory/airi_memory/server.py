"""FastAPI application exposing the memory protocol over HTTP REST."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import scoring
from .config import EmbeddingConfig, ServiceConfig, save_embedding_config
from .embedding import EmbeddingClient
from .store import MemoryStore, public_entry

SEARCH_BATCH_SIZE = 32


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    source_session_id: str | None = None


class MemoryPatchRequest(BaseModel):
    content: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)


class ConfigPayload(BaseModel):
    embedding: EmbeddingConfig


def _default_embedder_factory(embedding: EmbeddingConfig) -> EmbeddingClient:
    return EmbeddingClient(embedding.base_url, embedding.api_key, embedding.model)


class MemoryService:
    """Owns the store and the current embedding client; routes delegate here."""

    def __init__(
        self,
        config: ServiceConfig,
        store: MemoryStore,
        embedder_factory: Callable[[EmbeddingConfig], EmbeddingClient] = _default_embedder_factory,
    ) -> None:
        self.config = config
        self.store = store
        self._embedder_factory = embedder_factory
        self._embedder: EmbeddingClient | None = (
            embedder_factory(config.embedding) if config.embedding else None
        )

    @property
    def embedder(self) -> EmbeddingClient | None:
        return self._embedder

    def set_embedding_config(self, embedding: EmbeddingConfig) -> None:
        self.config.embedding = embedding
        self._embedder = self._embedder_factory(embedding)

    def require_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            raise HTTPException(
                status_code=503,
                detail='Embedding is not configured. Set it through PUT /v1/config first.',
            )
        return self._embedder

    async def backfill_missing_embeddings(self) -> int:
        """Embed entries stored before the embedding settings existed."""
        embedder = self.require_embedder()
        pending = self.store.entries_missing_embedding()
        backfilled = 0
        for start in range(0, len(pending), SEARCH_BATCH_SIZE):
            batch = pending[start:start + SEARCH_BATCH_SIZE]
            vectors = await embedder.embed_many([entry['content'] for entry in batch])
            for entry, vector in zip(batch, vectors):
                self.store.update_raw(entry['id'], lambda raw, vector=vector: {**raw, 'embedding': vector})
                backfilled += 1
        return backfilled

    async def create_memory(self, payload: MemoryCreateRequest) -> dict:
        embedder = self.require_embedder()
        vector = await embedder.embed(payload.content)
        duplicate = self.store.find_duplicate(vector)
        if duplicate is not None:
            now = time.time()
            reinforced = scoring.reinforce(
                {**duplicate, 'embedding': vector},
                now,
            )
            saved = self.store.replace(duplicate['id'], reinforced)
            return {'entry': saved, 'deduplicated': True}
        entry = self.store.add(payload.content, vector, payload.source_session_id)
        return {'entry': entry, 'deduplicated': False}

    async def search(self, payload: SearchRequest) -> dict:
        embedder = self.require_embedder()
        query_vector = await embedder.embed(payload.query)
        now = time.time()
        ranked = scoring.rank_entries(query_vector, self.store.raw_entries(), now)
        results = []
        for hit in ranked[:payload.top_k]:
            reinforced = scoring.reinforce(hit, now)
            # Ranking fields are per-request values; they must not leak into storage.
            persisted = {key: value for key, value in reinforced.items() if key not in ('score', 'similarity', 'time_relevance')}
            self.store.replace(hit['id'], persisted)
            results.append(public_entry(reinforced))
        return {'results': results}


def create_app(
    config: ServiceConfig,
    store: MemoryStore,
    embedder_factory: Callable[[EmbeddingConfig], EmbeddingClient] = _default_embedder_factory,
) -> FastAPI:
    service = MemoryService(config, store, embedder_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if service.embedder is not None:
            await service.embedder.close()

    app = FastAPI(title='airi-memory', lifespan=lifespan)

    # The service binds to 127.0.0.1 only; any local origin (web dev server,
    # Electron renderer) may call it, so the origin check stays permissive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    from . import __version__

    @app.get('/v1/health')
    async def health() -> dict:
        return {
            'status': 'ok',
            'version': __version__,
            'embedding_configured': service.config.embedding is not None,
        }

    @app.get('/v1/config')
    async def get_config() -> dict:
        return {'embedding': service.config.embedding}

    @app.put('/v1/config')
    async def put_config(payload: ConfigPayload) -> dict:
        service.set_embedding_config(payload.embedding)
        save_embedding_config(config.data_dir, payload.embedding)
        backfilled = await service.backfill_missing_embeddings()
        return {'embedding': service.config.embedding, 'backfilled': backfilled}

    @app.post('/v1/memories')
    async def create_memory(payload: MemoryCreateRequest) -> dict:
        return await service.create_memory(payload)

    @app.get('/v1/memories')
    async def list_memories(
        kind: str | None = Query(default=None, pattern='^(short|long)$'),
        q: str | None = None,
    ) -> dict:
        return {'entries': store.list_entries(kind=kind, q=q)}

    @app.patch('/v1/memories/{entry_id}')
    async def patch_memory(entry_id: str, payload: MemoryPatchRequest) -> dict:
        if store.get(entry_id) is None:
            raise HTTPException(status_code=404, detail=f'Memory {entry_id} not found')
        vector = None
        if service.embedder is not None:
            vector = await service.embedder.embed(payload.content)
        entry = store.update_content(entry_id, payload.content, vector)
        return {'entry': entry}

    @app.delete('/v1/memories/{entry_id}')
    async def delete_memory(entry_id: str) -> dict:
        if not store.delete(entry_id):
            raise HTTPException(status_code=404, detail=f'Memory {entry_id} not found')
        return {'deleted': entry_id}

    @app.delete('/v1/memories')
    async def clear_memories() -> dict:
        return {'deleted': store.clear()}

    @app.post('/v1/search')
    async def search_memories(payload: SearchRequest) -> dict:
        return await service.search(payload)

    return app
