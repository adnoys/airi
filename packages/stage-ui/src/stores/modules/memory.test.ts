// @vitest-environment jsdom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MemoryServiceClient } from '../../libs/memory/client'
import { DEFAULT_MEMORY_SERVICE_URL, parseExtractionResponse, useMemoryStore } from './memory'

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    locale: { value: 'en-US' },
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}))

vi.mock('../../libs/memory/client', () => ({
  MemoryServiceClient: vi.fn(),
  MemoryServiceError: class MemoryServiceError extends Error {},
}))

const { searchMock, healthMock, listMemoriesMock, deleteMemoryMock, patchMemoryMock } = vi.hoisted(() => ({
  searchMock: vi.fn(),
  healthMock: vi.fn(),
  listMemoriesMock: vi.fn(),
  deleteMemoryMock: vi.fn(),
  patchMemoryMock: vi.fn(),
}))

function stubClient() {
  vi.mocked(MemoryServiceClient).mockClear()
  // `new MemoryServiceClient()` requires a non-arrow implementation: arrow
  // functions cannot be constructed, and `new` on a mock honors that rule.
  vi.mocked(MemoryServiceClient).mockImplementation(function () {
    return {
      search: searchMock,
      health: healthMock,
      listMemories: listMemoriesMock,
      deleteMemory: deleteMemoryMock,
      patchMemory: patchMemoryMock,
    } as unknown as MemoryServiceClient
  } as unknown as typeof MemoryServiceClient)
}

describe('useMemoryStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    stubClient()
    searchMock.mockReset()
    healthMock.mockReset()
    listMemoriesMock.mockReset()
    deleteMemoryMock.mockReset()
    patchMemoryMock.mockReset()
  })

  describe('configured gate', () => {
    it('is false while disabled even with a service URL', () => {
      const store = useMemoryStore()
      store.enabled = false
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      expect(store.configured).toBe(false)
    })

    it('is false when the service URL is blank', () => {
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = '   '

      expect(store.configured).toBe(false)
    })

    it('is true when enabled with a service URL', () => {
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      expect(store.configured).toBe(true)
    })
  })

  describe('retrieveForQuery', () => {
    it('caches search results for the request snapshot', async () => {
      searchMock.mockResolvedValue([{ id: 'entry-1', content: 'User likes matcha', score: 1.2, similarity: 0.9, time_relevance: 0.5 }])
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      await store.retrieveForQuery('What does the user like?')

      expect(searchMock).toHaveBeenCalledWith('What does the user like?')
      expect(store.retrievedMemories).toHaveLength(1)
    })

    it('clears results and never throws when the service is unreachable', async () => {
      searchMock.mockRejectedValue(new Error('Failed to fetch'))
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      await expect(store.retrieveForQuery('Anything')).resolves.toBeUndefined()
      expect(store.retrievedMemories).toEqual([])
      expect(store.connection).toBe('unreachable')
    })

    it('does not call the service while disabled', async () => {
      const store = useMemoryStore()
      store.enabled = false
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      await store.retrieveForQuery('Anything')

      expect(searchMock).not.toHaveBeenCalled()
      expect(store.retrievedMemories).toEqual([])
    })
  })

  describe('reconcileRetrievedMemories', () => {
    it('deletes a memory the source user lines do not contain and hides it from this request', async () => {
      searchMock.mockResolvedValue([
        { id: 'yandere', content: '用户昨天问过病娇是什么。', source_session_id: 'session-1', score: 1, similarity: 1, time_relevance: 1 },
        { id: 'tea', content: '用户喜欢抹茶', source_session_id: 'session-1', score: 1, similarity: 1, time_relevance: 1 },
      ])
      listMemoriesMock.mockResolvedValue([
        { id: 'yandere', content: '用户昨天问过病娇是什么。', source_session_id: 'session-1' },
        { id: 'tea', content: '用户喜欢抹茶', source_session_id: 'session-1' },
      ])
      deleteMemoryMock.mockResolvedValue('yandere')
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL
      await store.retrieveForQuery('在乎')

      const revised = await store.reconcileRetrievedMemories(async () => ({
        userText: '我喜欢抹茶',
        assistantText: '病娇就是占有欲',
      }))

      expect(revised).toBe(1)
      expect(deleteMemoryMock).toHaveBeenCalledWith('yandere')
      expect(store.retrievedMemories.map(memory => memory.id)).toEqual(['tea'])
    })

    it('replaces a mixed memory with the matching user line', async () => {
      searchMock.mockResolvedValue([
        { id: 'mixed', content: '用户问过病娇，也喜欢抹茶', source_session_id: 'session-1', score: 1, similarity: 1, time_relevance: 1 },
      ])
      listMemoriesMock.mockResolvedValue([
        { id: 'mixed', content: '用户问过病娇，也喜欢抹茶', source_session_id: 'session-1' },
      ])
      patchMemoryMock.mockResolvedValue({ id: 'mixed', content: '用户原话：我喜欢抹茶' })
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL
      await store.retrieveForQuery('抹茶')

      const revised = await store.reconcileRetrievedMemories(async () => ({
        userText: '我喜欢抹茶',
        assistantText: '病娇是一种占有',
      }))

      expect(revised).toBe(1)
      expect(patchMemoryMock).toHaveBeenCalledWith('mixed', '用户原话：我喜欢抹茶')
      expect(store.retrievedMemories.map(memory => memory.content)).toEqual(['用户原话：我喜欢抹茶'])
    })

    it('does not read dialogues when a retrieved memory has no source session', async () => {
      searchMock.mockResolvedValue([
        { id: 'note', content: '用户喜欢抹茶', source_session_id: null, score: 1, similarity: 1, time_relevance: 1 },
      ])
      const store = useMemoryStore()
      store.enabled = true
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL
      await store.retrieveForQuery('抹茶')
      const loadDialogue = vi.fn()

      await expect(store.reconcileRetrievedMemories(loadDialogue)).resolves.toBe(0)
      expect(loadDialogue).not.toHaveBeenCalled()
      expect(listMemoriesMock).not.toHaveBeenCalled()
    })
  })

  describe('checkHealth', () => {
    it('reports connected when the service answers', async () => {
      healthMock.mockResolvedValue({ status: 'ok', version: '0.1.0', embedding_configured: true })
      const store = useMemoryStore()
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      await expect(store.checkHealth()).resolves.toBe(true)
      expect(store.connection).toBe('connected')
    })

    it('reports unreachable when the service fails', async () => {
      healthMock.mockRejectedValue(new Error('Failed to fetch'))
      const store = useMemoryStore()
      store.serviceUrl = DEFAULT_MEMORY_SERVICE_URL

      await expect(store.checkHealth()).resolves.toBe(false)
      expect(store.connection).toBe('unreachable')
    })
  })

  describe('parseExtractionResponse', () => {
    it('parses a plain JSON array', () => {
      expect(parseExtractionResponse('["User likes matcha","User owns a cat"]')).toEqual([
        'User likes matcha',
        'User owns a cat',
      ])
    })

    it('tolerates markdown fences around the array', () => {
      expect(parseExtractionResponse('```json\n["A fact"]\n```')).toEqual(['A fact'])
    })

    it('tolerates prose around the array', () => {
      expect(parseExtractionResponse('Here are the memories:\n["A fact"]\nDone.')).toEqual(['A fact'])
    })

    it('drops non-string items and blanks, and caps the list at five', () => {
      const raw = JSON.stringify(['a', 42, '  ', 'b', 'c', 'd', 'e', 'f'])
      expect(parseExtractionResponse(raw)).toEqual(['a', 'b', 'c', 'd', 'e'])
    })

    it('returns nothing for malformed output', () => {
      expect(parseExtractionResponse('not json at all')).toEqual([])
      expect(parseExtractionResponse('')).toEqual([])
    })
  })
})
