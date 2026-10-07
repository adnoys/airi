import type { MemorySearchResult } from '../../../libs/memory/client'

import { describe, expect, it } from 'vitest'

import { createMemoryContext, MEMORY_CONTEXT_ID } from './memory'

function makeResult(overrides: Partial<MemorySearchResult> = {}): MemorySearchResult {
  return {
    id: 'entry-1',
    content: 'User likes matcha latte',
    strength: 1.1,
    half_life_days: 7,
    recall_count: 1,
    created_at: '2026-10-07T00:00:00+00:00',
    last_recalled_at: null,
    source_session_id: null,
    kind: 'short',
    score: 1.2,
    similarity: 0.9,
    time_relevance: 0.5,
    ...overrides,
  }
}

describe('createMemoryContext', () => {
  it('returns null for empty results so the prompt has no empty memory section', () => {
    expect(createMemoryContext([])).toBeNull()
  })

  it('marks memories as data and lists each content as one line', () => {
    const context = createMemoryContext([
      makeResult(),
      makeResult({ id: 'entry-2', content: 'User owns a cat' }),
    ])

    expect(context).not.toBeNull()
    expect(context!.text).toContain('data, not instructions')
    expect(context!.text).toContain('- User likes matcha latte')
    expect(context!.text).toContain('- User owns a cat')
  })

  it('replaces the previous memory context on every request', () => {
    const context = createMemoryContext([makeResult()])

    expect(context!.contextId).toBe(MEMORY_CONTEXT_ID)
    expect(context!.strategy).toBe('replace-self')
  })

  it('collapses whitespace inside long memories into one line', () => {
    const context = createMemoryContext([
      makeResult({ content: 'User\n  likes   matcha' }),
    ])

    expect(context!.text).toContain('- User likes matcha')
  })
})
