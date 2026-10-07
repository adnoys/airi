"""Chat-model client used for LLM-based memory reranking.

DeepSeek does not serve embeddings, so in ``llm`` retrieval mode the
service ranks memories with keyword prefiltering plus a chat-model
rerank over the candidates. Any OpenAI-compatible chat endpoint works.
"""

from __future__ import annotations

import json

import httpx

RERANK_SYSTEM_PROMPT = (
    'You rank stored memories by how well they help answer the query. '
    'The user sends a query and a numbered list of memories. '
    'A query may ask about several things at once: pick at least one '
    'memory for every part of the query when a matching memory exists. '
    'Reply with a JSON array of the memory numbers, best match first, '
    'at most as many numbers as requested. '
    'Include a memory when any part of the query touches its topic, '
    'even if the wording is different. '
    'Reply with the numbers only, in a valid JSON array.'
)


class LLMError(RuntimeError):
    """Raised when the chat endpoint fails or returns an unusable answer."""


class LLMClient:
    """Minimal OpenAI-compatible chat client for reranking."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 60.0) -> None:
        self._endpoint = f'{base_url.rstrip("/")}/chat/completions'
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def rerank(self, query: str, candidates: list[str], top_k: int) -> list[int]:
        """Return candidate indices, most relevant first, at most ``top_k``.

        Tolerates prose or markdown fences around the JSON array and
        silently drops out-of-range or repeated indices.
        """
        listing = '\n'.join(f'{index}. {text}' for index, text in enumerate(candidates))
        response = await self._chat(
            RERANK_SYSTEM_PROMPT,
            f'QUERY: {query}\n\nREQUESTED: at most {top_k}\n\nMEMORIES:\n{listing}',
        )
        return self._parse_indices(response, len(candidates), top_k)

    async def _chat(self, system: str, user: str) -> str:
        try:
            response = await self._client.post(
                self._endpoint,
                json={
                    'model': self._model,
                    'messages': [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user},
                    ],
                    'temperature': 0,
                },
                headers={'Authorization': f'Bearer {self._api_key}'},
            )
            response.raise_for_status()
            body = response.json()
            content = body['choices'][0]['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise LLMError('Chat endpoint returned an empty message')
            return content
        except httpx.HTTPError as error:
            raise LLMError(f'Chat endpoint request failed: {error}') from error
        except (KeyError, TypeError, ValueError) as error:
            raise LLMError(f'Chat endpoint returned an unexpected body: {error}') from error

    @staticmethod
    def _parse_indices(raw: str, candidate_count: int, top_k: int) -> list[int]:
        text = raw.strip().replace('```json', '```').strip('`').strip()
        start = text.find('[')
        end = text.rfind(']')
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start:end + 1])
        except ValueError:
            return []
        if not isinstance(parsed, list):
            return []

        indices: list[int] = []
        for item in parsed:
            index = int(item) if isinstance(item, (int, float)) else -1
            if 0 <= index < candidate_count and index not in indices:
                indices.append(index)
            if len(indices) >= top_k:
                break
        return indices
