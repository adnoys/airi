"""JSON-file persistence for memory entries.

The whole collection lives in ``memories.json`` inside the data
directory. Writes replace the file atomically (tmp file + rename), so a
crash mid-write cannot corrupt the store.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import scoring

MEMORY_FILE_NAME = 'memories.json'


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class MemoryStore:
    """CRUD over the memory collection. Not process-safe; one service owns the file."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._file_path = data_dir / MEMORY_FILE_NAME
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._load()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def _load(self) -> None:
        if not self._file_path.exists():
            self._entries = []
            return
        with open(self._file_path, encoding='utf-8') as file:
            self._entries = json.load(file)

    def _save(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._file_path.with_suffix('.json.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as file:
            json.dump(self._entries, file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._file_path)

    def list_entries(self, kind: str | None = None, q: str | None = None) -> list[dict]:
        """Entries without embeddings, filtered by kind and a substring of content."""
        matched = self._entries
        if kind is not None:
            matched = [
                entry for entry in matched
                if scoring.memory_kind(float(entry.get('half_life_days', scoring.INITIAL_HALF_LIFE_DAYS))) == kind
            ]
        if q:
            needle = q.lower()
            matched = [entry for entry in matched if needle in entry['content'].lower()]
        return [public_entry(entry) for entry in matched]

    def get(self, entry_id: str) -> dict | None:
        for entry in self._entries:
            if entry['id'] == entry_id:
                return public_entry(entry)
        return None

    def raw_entries(self) -> list[dict]:
        """Every entry including embeddings; ranking needs the vectors."""
        return list(self._entries)

    def add(
        self,
        content: str,
        embedding: list[float] | None,
        source_session_id: str | None = None,
    ) -> dict:
        entry = {
            'id': uuid.uuid4().hex,
            'content': content,
            'embedding': embedding,
            'strength': scoring.INITIAL_STRENGTH,
            'half_life_days': scoring.INITIAL_HALF_LIFE_DAYS,
            'recall_count': 0,
            'created_at': utc_now_iso(),
            'last_recalled_at': None,
            'source_session_id': source_session_id,
        }
        with self._lock:
            self._entries.append(entry)
            self._save()
        return public_entry(entry)

    def find_duplicate(self, embedding: list[float]) -> dict | None:
        """The most similar stored entry above the dedup threshold, if any."""
        best: dict | None = None
        best_similarity = 0.0
        for entry in self._entries:
            stored = entry.get('embedding')
            if not stored:
                continue
            similarity = scoring.cosine_similarity(embedding, stored)
            if similarity >= scoring.DEDUP_SIMILARITY_THRESHOLD and similarity > best_similarity:
                best = entry
                best_similarity = similarity
        return best

    def update_raw(self, entry_id: str, transform) -> dict | None:
        """Apply ``transform`` to the raw entry in place and persist."""
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry['id'] != entry_id:
                    continue
                self._entries[index] = transform(entry)
                self._save()
                return public_entry(self._entries[index])
        return None

    def update_content(self, entry_id: str, content: str, embedding: list[float] | None) -> dict | None:
        return self.update_raw(entry_id, lambda entry: {**entry, 'content': content, 'embedding': embedding})

    def replace(self, entry_id: str, next_entry: dict) -> dict | None:
        return self.update_raw(entry_id, lambda _entry: next_entry)

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [entry for entry in self._entries if entry['id'] != entry_id]
            if len(self._entries) == before:
                return False
            self._save()
            return True

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries = []
            self._save()
            return count

    def entries_missing_embedding(self) -> list[dict]:
        return [entry for entry in self._entries if not entry.get('embedding')]


def public_entry(entry: dict) -> dict:
    """Strip the embedding vector and add the computed kind; raw vectors never leave the service."""
    public = {key: value for key, value in entry.items() if key != 'embedding'}
    public['kind'] = scoring.memory_kind(float(public.get('half_life_days', scoring.INITIAL_HALF_LIFE_DAYS)))
    return public
