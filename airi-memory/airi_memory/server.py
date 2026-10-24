"""FastAPI application exposing the memory protocol over HTTP REST.

Two retrieval modes share the same protocol:

- ``embedding``: memories are embedded through a remote OpenAI-compatible
  API and ranked by cosine similarity with time-relevance weighting.
- ``llm``: memories stay text-only; a keyword prefilter proposes
  candidates and a chat model (for example DeepSeek, which serves no
  embeddings) picks and orders the relevant ones.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import scoring
from .config import (
    RETRIEVAL_EMBEDDING,
    RETRIEVAL_LLM,
    EmbeddingConfig,
    LLMConfig,
    ServiceConfig,
    save_service_config,
)
from .embedding import EmbeddingClient
from .llm import LLMClient
from .store import MemoryStore, public_entry

SEARCH_BATCH_SIZE = 32
LLM_PREFILTER_LIMIT = 24
LLM_RANK_SCORE_STEP = 0.05


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    source_session_id: str | None = None


class MemoryPatchRequest(BaseModel):
    content: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)


class ConfigPayload(BaseModel):
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    retrieval: str = Field(default=RETRIEVAL_EMBEDDING, pattern=f'^({RETRIEVAL_EMBEDDING}|{RETRIEVAL_LLM})$')


def _default_embedder_factory(embedding: EmbeddingConfig) -> EmbeddingClient:
    return EmbeddingClient(embedding.base_url, embedding.api_key, embedding.model)


def _default_llm_factory(llm: LLMConfig) -> LLMClient:
    return LLMClient(llm.base_url, llm.api_key, llm.model)


def _redact(config: EmbeddingConfig | LLMConfig | None) -> dict | None:
    """Config without the key. Responses never echo secrets back."""
    if config is None:
        return None
    return config.model_dump() | {'api_key': ''}


def _with_kept_key(
    incoming: EmbeddingConfig | LLMConfig | None,
    stored: EmbeddingConfig | LLMConfig | None,
) -> EmbeddingConfig | LLMConfig | None:
    """Fill an empty incoming api_key from the stored config.

    GET never echoes keys, so the settings UI sends an empty key unless the
    user typed a new one. The stored key is kept only for the same endpoint
    and model; carrying an old key to a different provider would leak it
    there.
    """
    if incoming is None or incoming.api_key != '' or stored is None:
        return incoming
    if type(incoming) is not type(stored) or incoming.base_url != stored.base_url or incoming.model != stored.model:
        return incoming
    return incoming.model_copy(update={'api_key': stored.api_key})


class MemoryService:
    """Owns the store and the current retrieval clients; routes delegate here."""

    def __init__(
        self,
        config: ServiceConfig,
        store: MemoryStore,
        embedder_factory: Callable[[EmbeddingConfig], EmbeddingClient] = _default_embedder_factory,
        llm_factory: Callable[[LLMConfig], LLMClient] = _default_llm_factory,
    ) -> None:
        self.config = config
        self.store = store
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._embedder: EmbeddingClient | None = (
            embedder_factory(config.embedding) if config.embedding else None
        )
        self._llm: LLMClient | None = llm_factory(config.llm) if config.llm else None

    @property
    def embedder(self) -> EmbeddingClient | None:
        return self._embedder

    @property
    def llm(self) -> LLMClient | None:
        return self._llm

    @property
    def mode(self) -> str:
        return self.config.mode

    def set_config(self, embedding: EmbeddingConfig | None, llm: LLMConfig | None, retrieval: str) -> None:
        self.config.embedding = embedding
        self.config.llm = llm
        self.config.retrieval = retrieval
        self._embedder = self._embedder_factory(embedding) if embedding else None
        self._llm = self._llm_factory(llm) if llm else None

    def require_embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            raise HTTPException(
                status_code=503,
                detail='Embedding is not configured. Set it through PUT /v1/config first.',
            )
        return self._embedder

    def require_llm(self) -> LLMClient:
        if self._llm is None:
            raise HTTPException(
                status_code=503,
                detail='The reranking chat model is not configured. Set it through PUT /v1/config first.',
            )
        return self._llm

    async def backfill_missing_embeddings(self) -> int:
        """Embed entries stored before the embedding settings existed."""
        if self.mode != RETRIEVAL_EMBEDDING or self._embedder is None:
            return 0
        pending = self.store.entries_missing_embedding()
        backfilled = 0
        for start in range(0, len(pending), SEARCH_BATCH_SIZE):
            batch = pending[start:start + SEARCH_BATCH_SIZE]
            vectors = await self._embedder.embed_many([entry['content'] for entry in batch])
            backfilled += self.store.set_embeddings([(entry['id'], vector) for entry, vector in zip(batch, vectors)])
        return backfilled

    def require_mode(self) -> str:
        mode = self.mode
        if mode is None:
            raise HTTPException(
                status_code=503,
                detail='Retrieval is not configured. Set it through PUT /v1/config first.',
            )
        return mode

    async def create_memory(self, payload: MemoryCreateRequest) -> dict:
        if self.require_mode() == RETRIEVAL_EMBEDDING:
            embedder = self.require_embedder()
            vector = await embedder.embed(payload.content)
            duplicate = self.store.find_duplicate(vector)
        else:
            vector = None
            duplicate = self.store.find_duplicate_text(payload.content)

        if duplicate is not None:
            now = time.time()
            reinforced = scoring.reinforce(
                {**duplicate, 'embedding': vector} if vector else duplicate,
                now,
            )
            saved = self.store.replace(duplicate['id'], reinforced)
            return {'entry': saved, 'deduplicated': True}

        entry = self.store.add(payload.content, vector, payload.source_session_id)
        return {'entry': entry, 'deduplicated': False}

    async def search(self, payload: SearchRequest) -> dict:
        """Rank memories for one query.

        Search is read-only: it scores entries but never writes them
        back. Reinforcement happens only in the create path, when a
        restated fact deduplicates against an existing entry, so an
        automatic retrieval on every send cannot inflate half-lives.
        """
        now = time.time()

        if self.require_mode() == RETRIEVAL_EMBEDDING:
            embedder = self.require_embedder()
            query_vector = await embedder.embed(payload.query)
            ranked = scoring.rank_entries(query_vector, self.store.raw_entries(), now)
            return {'results': [public_entry(hit) for hit in ranked[:payload.top_k]]}

        llm = self.require_llm()
        entries = self.store.raw_entries()
        keyword_scores = scoring.keyword_scores(payload.query, entries)
        keyword_order = sorted(range(len(entries)), key=lambda index: keyword_scores[index], reverse=True)
        keyword_hits = [index for index in keyword_order if keyword_scores[index] > 0]
        # Keyword overlap alone cannot cover multi-part queries: a memory may
        # answer one part while sharing no wording with the query. Always pad
        # the candidate list with the most recent entries up to the limit, so
        # the reranking model sees them.
        recency_tail = list(reversed(range(len(entries))))
        candidate_indexes = list(dict.fromkeys(keyword_hits + recency_tail))[:LLM_PREFILTER_LIMIT]
        if not candidate_indexes:
            return {'results': []}

        candidates = [entries[index] for index in candidate_indexes]
        # Each line carries the stored local date, so the reranking model
        # answers day and date queries by ranking instead of a date shortcut
        # that bypasses relevance.
        candidate_lines = [
            f"[{scoring.entry_local_date(str(entry['created_at']))}] {str(entry.get('content', ''))}"
            for entry in candidates
        ]
        # The rerank client caps its answer; the slice keeps the top_k
        # contract for any rerank implementation.
        picked = (await llm.rerank(payload.query, candidate_lines, payload.top_k))[:payload.top_k]

        max_decayed = max(
            (
                scoring.decayed_strength(
                    float(entry.get('strength', scoring.INITIAL_STRENGTH)),
                    float(entry.get('half_life_days', scoring.INITIAL_HALF_LIFE_DAYS)),
                    scoring.age_in_days(entry['created_at'], now),
                )
                for entry in candidates
            ),
            default=0.0,
        )
        results = []
        for position, candidate_index in enumerate(picked):
            entry = candidates[candidate_index]
            similarity = max(0.0, 1.0 - position * LLM_RANK_SCORE_STEP)
            relevance = scoring.time_relevance(entry, now, max_decayed)
            score = scoring.SIMILARITY_WEIGHT * similarity + scoring.TIME_RELEVANCE_WEIGHT * relevance
            results.append(public_entry({**entry, 'score': score, 'similarity': similarity, 'time_relevance': relevance}))
        return {'results': results}


def create_app(
    config: ServiceConfig,
    store: MemoryStore,
    embedder_factory: Callable[[EmbeddingConfig], EmbeddingClient] = _default_embedder_factory,
    llm_factory: Callable[[LLMConfig], LLMClient] = _default_llm_factory,
) -> FastAPI:
    service = MemoryService(config, store, embedder_factory, llm_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if service.embedder is not None:
            await service.embedder.close()
        if service.llm is not None:
            await service.llm.close()

    app = FastAPI(title='airi-memory', lifespan=lifespan)

    # Only local origins may call the service: the web dev server and an
    # Electron renderer on localhost. A remote web page that reaches the
    # localhost port cannot read memories or the provider keys. There is no
    # authentication; any local process can still call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r'^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$',
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
            'llm_configured': service.config.llm is not None,
            'retrieval': service.mode,
        }

    @app.get('/v1/config')
    async def get_config() -> dict:
        return {
            'embedding': _redact(service.config.embedding),
            'llm': _redact(service.config.llm),
            'retrieval': service.config.retrieval,
        }

    @app.put('/v1/config')
    async def put_config(payload: ConfigPayload) -> dict:
        # An omitted client keeps its current settings: the settings page
        # sends only the client that matches the selected mode.
        embedding = _with_kept_key(payload.embedding, service.config.embedding) if payload.embedding is not None else service.config.embedding
        llm = _with_kept_key(payload.llm, service.config.llm) if payload.llm is not None else service.config.llm
        service.set_config(embedding, llm, payload.retrieval)
        save_service_config(config.data_dir, embedding, llm, payload.retrieval)
        backfilled = await service.backfill_missing_embeddings()
        return {
            'embedding': _redact(embedding),
            'llm': _redact(llm),
            'retrieval': service.config.retrieval,
            'backfilled': backfilled,
        }

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
        if service.mode == RETRIEVAL_EMBEDDING and service.embedder is not None:
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
