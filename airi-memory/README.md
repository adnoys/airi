# airi-memory

A standalone memory service for AIRI. It stores memories as JSON, embeds them through a remote OpenAI-compatible API, and serves recall, decay, and reinforcement over HTTP REST. AIRI connects to it as a plain HTTP client.

## What it does

- Stores memory entries with content, an embedding vector, a half-life, a strength, and recall counters.
- Embeds content through a remote OpenAI-compatible `/embeddings` endpoint, so no model runs locally.
- Ranks memories for a query by `1.2 * similarity + 0.2 * time_relevance`, where time relevance comes from a stateless half-life decay. No background job ever rewrites scores.
- Reinforces memories when they are recalled: recall count grows, strength rises, and the half-life lengthens. Long-lived memories cross a half-life threshold and become `long` term. This mirrors the decay design in the AIRI memory devlogs (`docs/content/en/blog/DevLog-2025.04.14`).
- Deduplicates writes: a new memory with cosine similarity of at least 0.92 against an existing entry reinforces that entry instead of creating a copy.

## How to run

All dependencies live in the `airi` conda environment (no separate venv):

```sh
conda install -n airi python=3.12 -y
conda run -n airi python -m pip install -r requirements.txt
```

Start the service from this directory:

```sh
conda run -n airi python -m airi_memory
```

The service listens on `http://127.0.0.1:6430` and binds to localhost only. Data lives in `data/memories.json`; embedding settings live in `data/config.json`. Both directories are gitignored.

Flags and environment variables:

| Flag | Environment | Default | Purpose |
| --- | --- | --- | --- |
| `--host` | `AIRI_MEMORY_HOST` | `127.0.0.1` | Bind address |
| `--port` | `AIRI_MEMORY_PORT` | `6430` | Bind port |
| `--data-dir` | `AIRI_MEMORY_DATA_DIR` | `./data` | Data directory |
| — | `AIRI_MEMORY_EMBED_BASE_URL` | — | Embedding API root |
| — | `AIRI_MEMORY_EMBED_API_KEY` | — | Embedding API key |
| — | `AIRI_MEMORY_EMBED_MODEL` | — | Embedding model name |

You can also set the embedding settings at runtime from the AIRI memory settings page, which calls `PUT /v1/config`.

## Protocol

All endpoints are JSON under `/v1`:

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/health` | Liveness and whether embedding is configured |
| `GET /v1/config`, `PUT /v1/config` | Read or set the embedding settings; `PUT` backfills embeddings for entries stored without one |
| `POST /v1/memories` | Store a memory; deduplicates and reinforces when similar content exists |
| `GET /v1/memories?kind=short\|long&q=` | List memories, optionally filtered by kind and a content substring |
| `PATCH /v1/memories/{id}` | Edit content and re-embed |
| `DELETE /v1/memories/{id}` | Delete one memory |
| `DELETE /v1/memories` | Clear all memories |
| `POST /v1/search` | `{ "query": "...", "top_k": 6 }`; returns ranked memories and reinforces the hits |

Memory entries carry `id`, `content`, `strength`, `half_life_days`, `recall_count`, `created_at`, `last_recalled_at`, `source_session_id`, and a computed `kind`. Raw embedding vectors never leave the service.

## Tests

```sh
conda run -n airi python -m pytest tests -v
```

## When not to use it

- Do not expose it to a network. It has no authentication and trusts every caller on the local machine.
- It is a single-process, single-user service. It does not shard, replicate, or sync across devices.
- For a multi-tenant or hosted deployment, build on a real vector store instead.
