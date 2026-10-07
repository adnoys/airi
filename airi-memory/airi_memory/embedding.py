"""Remote embedding client for OpenAI-compatible APIs."""

from __future__ import annotations

import httpx


class EmbeddingError(RuntimeError):
    """Raised when the remote embedding API fails or returns an unexpected body."""


class EmbeddingClient:
    """Embeds text through ``{base_url}/embeddings`` (OpenAI-compatible).

    ``base_url`` is the API root, for example ``https://api.openai.com/v1``.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 30.0) -> None:
        self._endpoint = f'{base_url.rstrip("/")}/embeddings'
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one request, in input order."""
        if not texts:
            return []
        try:
            response = await self._client.post(
                self._endpoint,
                json={'model': self._model, 'input': texts},
                headers={'Authorization': f'Bearer {self._api_key}'},
            )
            response.raise_for_status()
            body = response.json()
            data = body.get('data')
            if not isinstance(data, list) or len(data) != len(texts):
                raise EmbeddingError(
                    f'Embedding API returned {len(data) if isinstance(data, list) else "no"} vectors for {len(texts)} inputs',
                )
            ordered = sorted(data, key=lambda item: item.get('index', 0))
            return [item['embedding'] for item in ordered]
        except httpx.HTTPError as error:
            raise EmbeddingError(f'Embedding API request failed: {error}') from error
        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingError(f'Embedding API returned an unexpected body: {error}') from error

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        return vectors[0]

    async def close(self) -> None:
        await self._client.aclose()
