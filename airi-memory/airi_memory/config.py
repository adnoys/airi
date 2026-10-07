"""Service configuration.

The embedding settings live in ``data/config.json`` (written by the
``PUT /v1/config`` endpoint or edited by hand). Environment variables
override the file, and the bind address comes from the command line or
environment with localhost defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

CONFIG_FILE_NAME = 'config.json'

ENV_DATA_DIR = 'AIRI_MEMORY_DATA_DIR'
ENV_HOST = 'AIRI_MEMORY_HOST'
ENV_PORT = 'AIRI_MEMORY_PORT'
ENV_EMBED_BASE_URL = 'AIRI_MEMORY_EMBED_BASE_URL'
ENV_EMBED_API_KEY = 'AIRI_MEMORY_EMBED_API_KEY'
ENV_EMBED_MODEL = 'AIRI_MEMORY_EMBED_MODEL'

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6430


class EmbeddingConfig(BaseModel):
    base_url: str
    api_key: str = ''
    model: str


class ServiceConfig(BaseModel):
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    data_dir: Path
    embedding: EmbeddingConfig | None = None


def default_data_dir() -> Path:
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / 'data'


def load_config(data_dir: Path | None = None) -> ServiceConfig:
    """Load configuration; the environment wins over ``config.json``."""
    data_dir = data_dir or default_data_dir()
    config_path = data_dir / CONFIG_FILE_NAME

    embedding: EmbeddingConfig | None = None
    if config_path.exists():
        with open(config_path, encoding='utf-8') as file:
            raw = json.load(file)
        if isinstance(raw.get('embedding'), dict):
            embedding = EmbeddingConfig.model_validate(raw['embedding'])

    if os.environ.get(ENV_EMBED_BASE_URL) and os.environ.get(ENV_EMBED_MODEL):
        embedding = EmbeddingConfig(
            base_url=os.environ[ENV_EMBED_BASE_URL],
            api_key=os.environ.get(ENV_EMBED_API_KEY, ''),
            model=os.environ[ENV_EMBED_MODEL],
        )

    host = os.environ.get(ENV_HOST, DEFAULT_HOST)
    port = int(os.environ.get(ENV_PORT, DEFAULT_PORT))
    return ServiceConfig(host=host, port=port, data_dir=data_dir, embedding=embedding)


def save_embedding_config(data_dir: Path, embedding: EmbeddingConfig) -> None:
    """Persist the embedding settings; other fields stay untouched."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / CONFIG_FILE_NAME
    raw: dict = {}
    if config_path.exists():
        with open(config_path, encoding='utf-8') as file:
            raw = json.load(file)
    raw['embedding'] = embedding.model_dump()
    tmp_path = config_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(raw, file, ensure_ascii=False, indent=2)
    tmp_path.replace(config_path)
