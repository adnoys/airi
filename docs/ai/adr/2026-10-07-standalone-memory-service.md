# Memory lives in a standalone local service behind an HTTP protocol

Status: Accepted

## Context

AIRI needs long-term memory across chat sessions. The design direction comes
from the memory devlogs (`docs/content/en/blog/DevLog-2025.04.06` and
`DevLog-2025.04.14`): memories decay by a half-life, recall reinforces them,
and ranking combines vector similarity with time relevance. Mem0 and Zep were
evaluated in the devlogs and rejected for role-play and companion use.

The first implementation draft put the whole memory system inside
`packages/stage-ui`: an IndexedDB repository, an embedding client, scoring
functions, and the retrieval pipeline all as in-process TypeScript. That
couples memory storage to the renderer bundle and makes the memory data hard
to inspect, back up, or reuse outside the app.

## Decision

- Memory is a standalone service in `airi-memory/` at the repo root, written
  in Python (FastAPI), run from the `airi` conda environment.
- AIRI talks to it over HTTP REST on `127.0.0.1:6430`. The protocol is the
  contract: health, config, memory CRUD, and search with reinforcement.
- The service owns storage (JSON file), embeddings (remote OpenAI-compatible
  API), scoring (stateless decay), deduplication, and reinforcement. AIRI owns
  extraction (it already holds a configured chat model), injection into the
  chat context, and all settings UI.
- Retrieval happens during send preparation and rides into the request
  snapshot like the account context. It never persists in the context
  registry, so stale memories cannot leak into later requests.
- The service binds to localhost only, has no authentication, and stores data
  in a gitignored `data/` directory.

Alternatives considered:

- In-process TypeScript module in `stage-ui`: simplest to ship, but memory data
  becomes invisible outside the app, and the renderer owns a data domain that
  outlives renderer sessions.
- MCP server: the protocol of choice for tools, but AIRI has no MCP client in
  the renderer yet, and memories are not a tool call — they are context that
  must arrive before prompt composition on every turn.

## Consequences

- The service must run before memory works. AIRI degrades gracefully: a
  failing service yields no memories and never fails a chat send.
- Extraction costs one extra chat-model call per turn when auto-extract is
  on. It defaults to off.
- Embedding costs one API call per query and per stored memory. The API key
  lives in the service config, not in the AIRI provider store.
- `packages/memory-pgvector` stays an unused skeleton. A server-hosted memory
  backend would need a different decision.
