import { describe, expect, it, vi } from 'vitest'

import { MemoryServiceClient, MemoryServiceError } from './client'

function respond(body: unknown, status = 200) {
  // A fresh Response per call: a Response body can be read only once.
  return vi.fn().mockImplementation(() => Promise.resolve(new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })))
}

function makeEntry(id = 'entry-1') {
  return {
    id,
    content: 'User likes matcha',
    strength: 1,
    half_life_days: 7,
    recall_count: 0,
    created_at: '2026-10-07T00:00:00+00:00',
    last_recalled_at: null,
    source_session_id: null,
    kind: 'short',
  }
}

describe('memory service client', () => {
  it('normalizes a trailing slash on the base URL', async () => {
    const fetcher = respond({ results: [] })
    const client = new MemoryServiceClient('http://127.0.0.1:6430///', fetcher as unknown as typeof fetch)

    await client.search('hello')

    expect(fetcher).toHaveBeenCalledWith('http://127.0.0.1:6430/v1/search', expect.anything())
  })

  it('posts the query and top_k to /v1/search', async () => {
    const fetcher = respond({ results: [] })
    const client = new MemoryServiceClient('http://127.0.0.1:6430', fetcher as unknown as typeof fetch)

    await client.search('hello', 3)

    const [, init] = fetcher.mock.calls[0] as [string, RequestInit]
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual({ query: 'hello', top_k: 3 })
  })

  it('builds kind and text filters into the list query string', async () => {
    const fetcher = respond({ entries: [] })
    const client = new MemoryServiceClient('http://127.0.0.1:6430', fetcher as unknown as typeof fetch)

    await client.listMemories('long', 'matcha')

    expect(fetcher).toHaveBeenCalledWith('http://127.0.0.1:6430/v1/memories?kind=long&q=matcha', expect.anything())
  })

  it('validates entry fields of a search response', async () => {
    const entry = makeEntry()
    const fetcher = respond({ results: [{ ...entry, score: 1.2, similarity: 0.9, time_relevance: 0.5 }] })
    const client = new MemoryServiceClient('http://127.0.0.1:6430', fetcher as unknown as typeof fetch)

    const results = await client.search('hello')

    expect(results).toHaveLength(1)
    expect(results[0].content).toBe('User likes matcha')
    expect(results[0].score).toBe(1.2)
  })

  it('surfaces the FastAPI detail message on protocol errors', async () => {
    const fetcher = respond({ detail: 'Embedding is not configured.' }, 503)
    const client = new MemoryServiceClient('http://127.0.0.1:6430', fetcher as unknown as typeof fetch)

    await expect(client.createMemory('Anything')).rejects.toThrow(MemoryServiceError)
    await expect(client.createMemory('Anything')).rejects.toThrow('Embedding is not configured.')
  })

  it('wraps network failures as unreachable', async () => {
    const fetcher = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))
    const client = new MemoryServiceClient('http://127.0.0.1:6430', fetcher as unknown as typeof fetch)

    await expect(client.health()).rejects.toThrow('unreachable')
  })
})
