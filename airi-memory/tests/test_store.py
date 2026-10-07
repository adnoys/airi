"""Tests for the JSON-file memory store."""

from __future__ import annotations

from pathlib import Path

import pytest

from airi_memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


class TestCrud:
    def test_add_and_list_roundtrip(self, store: MemoryStore) -> None:
        entry = store.add('User likes matcha', [1.0, 0.0], 'session-1')
        entries = store.list_entries()
        assert len(entries) == 1
        assert entries[0]['content'] == 'User likes matcha'
        assert entries[0]['source_session_id'] == 'session-1'
        assert 'embedding' not in entries[0]
        assert entry['id']

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        first = MemoryStore(tmp_path)
        entry = first.add('Persisted memory', [1.0, 0.0])
        second = MemoryStore(tmp_path)
        assert second.get(entry['id']) == entry

    def test_get_unknown_returns_none(self, store: MemoryStore) -> None:
        assert store.get('missing') is None

    def test_update_content(self, store: MemoryStore) -> None:
        entry = store.add('Old content', [1.0, 0.0])
        updated = store.update_content(entry['id'], 'New content', [0.0, 1.0])
        assert updated is not None
        assert updated['content'] == 'New content'
        assert store.get(entry['id'])['content'] == 'New content'

    def test_replace(self, store: MemoryStore) -> None:
        entry = store.add('Original', [1.0, 0.0])
        replaced = store.replace(entry['id'], {**entry, 'recall_count': 5})
        assert replaced['recall_count'] == 5

    def test_delete(self, store: MemoryStore) -> None:
        entry = store.add('To delete', [1.0, 0.0])
        assert store.delete(entry['id']) is True
        assert store.delete(entry['id']) is False
        assert store.list_entries() == []

    def test_clear(self, store: MemoryStore) -> None:
        store.add('One', [1.0, 0.0])
        store.add('Two', [0.0, 1.0])
        assert store.clear() == 2
        assert store.list_entries() == []


class TestFilters:
    def test_filter_by_kind(self, store: MemoryStore) -> None:
        short = store.add('Short memory', [1.0, 0.0])
        long_entry = store.add('Long memory', [0.0, 1.0])
        store.replace(long_entry['id'], {**long_entry, 'half_life_days': 30.0})
        shorts = store.list_entries(kind='short')
        longs = store.list_entries(kind='long')
        assert [entry['id'] for entry in shorts] == [short['id']]
        assert [entry['id'] for entry in longs] == [long_entry['id']]

    def test_filter_by_substring(self, store: MemoryStore) -> None:
        store.add('User likes matcha latte', [1.0, 0.0])
        store.add('User owns a cat', [0.0, 1.0])
        matched = store.list_entries(q='MATCHA')
        assert len(matched) == 1
        assert matched[0]['content'] == 'User likes matcha latte'


class TestDuplicates:
    def test_find_duplicate_above_threshold(self, store: MemoryStore) -> None:
        store.add('User likes matcha', [1.0, 0.0])
        duplicate = store.find_duplicate([0.99, 0.1])
        assert duplicate is not None
        assert duplicate['content'] == 'User likes matcha'

    def test_unrelated_vector_is_not_duplicate(self, store: MemoryStore) -> None:
        store.add('User likes matcha', [1.0, 0.0])
        assert store.find_duplicate([0.0, 1.0]) is None

    def test_entries_without_embedding_are_ignored(self, store: MemoryStore) -> None:
        store.add('Never embedded', None)
        assert store.find_duplicate([1.0, 0.0]) is None

    def test_entries_missing_embedding_listing(self, store: MemoryStore) -> None:
        store.add('Embedded', [1.0, 0.0])
        store.add('Missing', None)
        pending = store.entries_missing_embedding()
        assert [entry['content'] for entry in pending] == ['Missing']
