# Memory search is read-only and the reranker sees dates

Status: Accepted

## Context

The memory service from [2026-10-07-standalone-memory-service.md](2026-10-07-standalone-memory-service.md) retrieved memories on every send. Three properties of that pipeline damaged it:

- Search reinforced every returned entry. With one search per message, any entry in the top six gained strength and a longer half-life per message. One evening of chat made the recalled entries permanent. The recall signal stopped meaning that a memory mattered.
- The renderer re-verified retrieved memories against their source dialogues before the model read them. That step sat on the send path and could not afford a model call, so it used string heuristics. A memory whose words differed from its source line was deleted silently. Paraphrased memories died on first retrieval.
- The reranking model received candidates without dates, so a query naming a day could not be answered by ranking. A date shortcut in front of ranking caught every query containing 今天 or 昨天 and returned every entry of that day, unreinforced relevance aside. It also reinforced all of them.

`GET /v1/config` also echoed provider keys, and CORS allowed every origin, so any web page open in the browser could read the keys and the memories.

## Decision

- Search is read-only. `POST /v1/search` scores entries and returns them. It never writes.
- Reinforcement has one signal: the create path deduplicates a restated fact and reinforces the stored entry. A memory fades unless the user states the fact again.
- Data quality is a write-time concern. The extraction prompt requires each fact to end with the speaker's verbatim words. The read-time revision layer is deleted.
- Rerank candidates carry the stored local date, for example `[2026-10-24]`. The reranking prompt tells the model to use the dates. Date and day queries are answered by ranking, so no date shortcut exists.
- `GET /v1/config` and the `PUT` response never echo keys. An empty key on `PUT` keeps the stored key when the endpoint and model are unchanged. An omitted client keeps its settings, because the settings page sends only the client that matches the selected mode.
- CORS allows localhost origins only. There is still no authentication: any local process can call the API.
- Data lives in the platform application-data directory. First start copies an existing in-repo `data/` directory forward once, so a repository clean can no longer wipe memories.
- Text deduplication compares the fact before the parenthetical quote. A restated fact with a freshly chosen quote reinforces the stored entry instead of creating a near-duplicate.

## Consequences

- `recall_count` counts re-statements of a fact, not retrievals.
- Day and date recall depends on the reranking model reading the dates. A weak model can miss them.
- Extraction quality decides memory quality. A wrong fact stays until the user deletes it or states the truth again. The verbatim quote makes wrong facts visible in the settings list.
- The settings page shows an empty key field after reload. Saving without typing a key keeps the stored key for the same endpoint and model.
- The send path performs one search request and nothing else. In `llm` retrieval mode that request still contains one reranking model call.
