"""Service configuration.

The embedding settings live in ``data/config.json`` (written by the
``PUT /v1/config`` endpoint or edited by hand). Environment variables
override the file, and the bind address comes from the command line or
environment with localhost defaults.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from pydantic import BaseModel

from .store import MEMORY_FILE_NAME

CONFIG_FILE_NAME = 'config.json'

ENV_DATA_DIR = 'AIRI_MEMORY_DATA_DIR'
ENV_HOST = 'AIRI_MEMORY_HOST'
ENV_PORT = 'AIRI_MEMORY_PORT'
ENV_EMBED_BASE_URL = 'AIRI_MEMORY_EMBED_BASE_URL'
ENV_EMBED_API_KEY = 'AIRI_MEMORY_EMBED_API_KEY'
ENV_EMBED_MODEL = 'AIRI_MEMORY_EMBED_MODEL'
ENV_LLM_BASE_URL = 'AIRI_MEMORY_LLM_BASE_URL'
ENV_LLM_API_KEY = 'AIRI_MEMORY_LLM_API_KEY'
ENV_LLM_MODEL = 'AIRI_MEMORY_LLM_MODEL'

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6430

RETRIEVAL_EMBEDDING = 'embedding'
RETRIEVAL_LLM = 'llm'


class EmbeddingConfig(BaseModel):
    base_url: str
    api_key: str = ''
    model: str


class LLMConfig(BaseModel):
    """Chat endpoint used to rerank memories in ``llm`` retrieval mode."""

    base_url: str
    api_key: str = ''
    model: str


class ServiceConfig(BaseModel):
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    data_dir: Path
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    retrieval: str = RETRIEVAL_EMBEDDING

    @property
    def mode(self) -> str | None:
        """The retrieval mode that can run with the clients configured now.

        The preferred mode wins when its client exists. Otherwise the other
        configured client is used. None means neither client exists.
        """
        if self.retrieval == RETRIEVAL_LLM:
            if self.llm is not None:
                return RETRIEVAL_LLM
            if self.embedding is not None:
                return RETRIEVAL_EMBEDDING
            return None
        if self.embedding is not None:
            return RETRIEVAL_EMBEDDING
        if self.llm is not None:
            return RETRIEVAL_LLM
        return None


def platform_data_dir() -> Path:
    """Per-user application data directory, outside any repository."""
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'airi-memory'
    if sys.platform == 'win32':
        return Path.home() / 'AppData' / 'Local' / 'airi-memory'
    return Path.home() / '.local' / 'share' / 'airi-memory'


def legacy_data_dir() -> Path:
    """The data directory inside the repository, used before 2026-10-24."""
    return Path(__file__).resolve().parent.parent / 'data'


def migrate_legacy_data(target: Path) -> bool:
    """Copy an in-repo data directory forward, once.

    Memories used to live inside the repository, where a clean checkout
    wiped them. Only files missing at the target are copied, so the
    target always wins afterwards.
    """
    legacy = legacy_data_dir()
    try:
        same = legacy.resolve() == target.resolve()
    except OSError:
        same = False
    if same or not legacy.is_dir():
        return False
    copied = False
    for file_name in (MEMORY_FILE_NAME, CONFIG_FILE_NAME):
        source = legacy / file_name
        destination = target / file_name
        if source.is_file() and not destination.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied = True
    if copied:
        print(f'airi-memory: copied existing data from {legacy} to {target}')
    return copied


def default_data_dir() -> Path:
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override)
    return platform_data_dir()


def load_config(data_dir: Path | None = None) -> ServiceConfig:
    """Load configuration; the environment wins over ``config.json``.

    Without an explicit data directory the platform default is used and
    an in-repo legacy data directory is copied forward once.
    """
    explicit = data_dir is not None or bool(os.environ.get(ENV_DATA_DIR))
    data_dir = data_dir or default_data_dir()
    if not explicit:
        migrate_legacy_data(data_dir)
    config_path = data_dir / CONFIG_FILE_NAME

    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    retrieval = RETRIEVAL_EMBEDDING
    if config_path.exists():
        with open(config_path, encoding='utf-8') as file:
            raw = json.load(file)
        if isinstance(raw.get('embedding'), dict):
            embedding = EmbeddingConfig.model_validate(raw['embedding'])
        if isinstance(raw.get('llm'), dict):
            llm = LLMConfig.model_validate(raw['llm'])
        if raw.get('retrieval') in (RETRIEVAL_EMBEDDING, RETRIEVAL_LLM):
            retrieval = raw['retrieval']

    if os.environ.get(ENV_EMBED_BASE_URL) and os.environ.get(ENV_EMBED_MODEL):
        embedding = EmbeddingConfig(
            base_url=os.environ[ENV_EMBED_BASE_URL],
            api_key=os.environ.get(ENV_EMBED_API_KEY, ''),
            model=os.environ[ENV_EMBED_MODEL],
        )

    if os.environ.get(ENV_LLM_BASE_URL) and os.environ.get(ENV_LLM_MODEL):
        llm = LLMConfig(
            base_url=os.environ[ENV_LLM_BASE_URL],
            api_key=os.environ.get(ENV_LLM_API_KEY, ''),
            model=os.environ[ENV_LLM_MODEL],
        )

    host = os.environ.get(ENV_HOST, DEFAULT_HOST)
    port = int(os.environ.get(ENV_PORT, DEFAULT_PORT))
    return ServiceConfig(host=host, port=port, data_dir=data_dir, embedding=embedding, llm=llm, retrieval=retrieval)


def save_service_config(data_dir: Path, embedding: EmbeddingConfig | None, llm: LLMConfig | None, retrieval: str) -> None:
    """Persist the retrieval settings; other fields stay untouched."""
    data_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / CONFIG_FILE_NAME
    raw: dict = {}
    if config_path.exists():
        with open(config_path, encoding='utf-8') as file:
            raw = json.load(file)
    if embedding is not None:
        raw['embedding'] = embedding.model_dump()
    if llm is not None:
        raw['llm'] = llm.model_dump()
    raw['retrieval'] = retrieval
    tmp_path = config_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(raw, file, ensure_ascii=False, indent=2)
    tmp_path.replace(config_path)
