"""Tests for config load, save, and the usable retrieval mode."""

from __future__ import annotations

import json
from pathlib import Path

from airi_memory.config import (
    ENV_DATA_DIR,
    ENV_EMBED_API_KEY,
    ENV_EMBED_BASE_URL,
    ENV_EMBED_MODEL,
    ENV_HOST,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_PORT,
    RETRIEVAL_EMBEDDING,
    RETRIEVAL_LLM,
    EmbeddingConfig,
    LLMConfig,
    ServiceConfig,
    default_data_dir,
    load_config,
    save_service_config,
)


def _config(tmp_path: Path, retrieval: str, embedding: bool, llm: bool) -> ServiceConfig:
    return ServiceConfig(
        data_dir=tmp_path,
        retrieval=retrieval,
        embedding=EmbeddingConfig(base_url='https://embed.example', model='embed') if embedding else None,
        llm=LLMConfig(base_url='https://llm.example', model='chat') if llm else None,
    )


class TestMode:
    def test_embedding_mode_uses_the_embedding_client(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_EMBEDDING, True, False).mode == RETRIEVAL_EMBEDDING

    def test_embedding_mode_falls_back_to_the_chat_model(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_EMBEDDING, False, True).mode == RETRIEVAL_LLM

    def test_embedding_mode_stays_embedding_without_either_client(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_EMBEDDING, False, False).mode == RETRIEVAL_EMBEDDING

    def test_llm_mode_uses_the_chat_model(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_LLM, False, True).mode == RETRIEVAL_LLM

    def test_llm_mode_falls_back_to_embeddings(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_LLM, True, False).mode == RETRIEVAL_EMBEDDING

    def test_llm_mode_stays_llm_without_either_client(self, tmp_path: Path) -> None:
        assert _config(tmp_path, RETRIEVAL_LLM, False, False).mode == RETRIEVAL_LLM


class TestLoadConfig:
    def test_reads_retrieval_settings_from_the_file(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv(ENV_EMBED_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_EMBED_MODEL, raising=False)
        monkeypatch.delenv(ENV_LLM_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_LLM_MODEL, raising=False)
        (tmp_path / 'config.json').write_text(json.dumps({
            'embedding': {'base_url': 'https://embed.example', 'api_key': 'file-key', 'model': 'file-embed'},
            'llm': {'base_url': 'https://llm.example', 'api_key': 'llm-key', 'model': 'file-chat'},
            'retrieval': 'llm',
        }), encoding='utf-8')

        config = load_config(tmp_path)

        assert config.retrieval == RETRIEVAL_LLM
        assert config.embedding is not None
        assert config.embedding.model == 'file-embed'
        assert config.llm is not None
        assert config.llm.model == 'file-chat'

    def test_environment_overrides_a_complete_embedding_pair(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / 'config.json').write_text(json.dumps({
            'embedding': {'base_url': 'https://file.example', 'model': 'file-model'},
        }), encoding='utf-8')
        monkeypatch.setenv(ENV_EMBED_BASE_URL, 'https://env.example')
        monkeypatch.setenv(ENV_EMBED_MODEL, 'env-model')
        monkeypatch.setenv(ENV_EMBED_API_KEY, 'env-key')

        embedding = load_config(tmp_path).embedding

        assert embedding is not None
        assert embedding.base_url == 'https://env.example'
        assert embedding.model == 'env-model'
        assert embedding.api_key == 'env-key'

    def test_partial_embedding_environment_keeps_the_file(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / 'config.json').write_text(json.dumps({
            'embedding': {'base_url': 'https://file.example', 'model': 'file-model'},
        }), encoding='utf-8')
        monkeypatch.delenv(ENV_EMBED_BASE_URL, raising=False)
        monkeypatch.setenv(ENV_EMBED_MODEL, 'env-model')

        embedding = load_config(tmp_path).embedding

        assert embedding is not None
        assert embedding.model == 'file-model'

    def test_ignores_an_unknown_retrieval_value(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv(ENV_LLM_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_LLM_MODEL, raising=False)
        (tmp_path / 'config.json').write_text(json.dumps({'retrieval': 'other'}), encoding='utf-8')

        assert load_config(tmp_path).retrieval == RETRIEVAL_EMBEDDING

    def test_host_and_port_come_from_the_environment(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(ENV_HOST, '0.0.0.0')
        monkeypatch.setenv(ENV_PORT, '7001')

        config = load_config(tmp_path)

        assert config.host == '0.0.0.0'
        assert config.port == 7001

    def test_data_dir_environment_overrides_the_default(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))

        assert default_data_dir() == tmp_path


class TestSaveServiceConfig:
    def test_keeps_unrelated_fields(self, tmp_path: Path) -> None:
        (tmp_path / 'config.json').write_text(json.dumps({'note': 'keep'}), encoding='utf-8')

        save_service_config(
            tmp_path,
            EmbeddingConfig(base_url='https://embed.example', model='embed'),
            None,
            RETRIEVAL_LLM,
        )

        raw = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
        assert raw['note'] == 'keep'
        assert raw['retrieval'] == RETRIEVAL_LLM
        assert raw['embedding']['model'] == 'embed'
        assert 'llm' not in raw
