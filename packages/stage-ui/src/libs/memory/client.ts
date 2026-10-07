import type { InferOutput } from 'valibot'

import { errorMessageFrom } from '@moeru/std'
import { array, boolean, literal, null_, nullable, number, object, parse, string, union } from 'valibot'

/**
 * HTTP client for the airi-memory service (see `airi-memory/` at the repo root).
 *
 * The service owns storage, embeddings, decay, and reinforcement; this client
 * only speaks the REST protocol. Responses are validated with Valibot so a
 * mismatched service version fails loudly instead of leaking bad data into
 * the chat pipeline.
 */

const memoryKindSchema = union([literal('short'), literal('long')])

const memoryEntrySchema = object({
  id: string(),
  content: string(),
  strength: number(),
  half_life_days: number(),
  recall_count: number(),
  created_at: string(),
  last_recalled_at: nullable(string()),
  source_session_id: nullable(string()),
  kind: memoryKindSchema,
})

const memorySearchResultSchema = object({
  ...memoryEntrySchema.entries,
  score: number(),
  similarity: number(),
  time_relevance: number(),
})

const healthSchema = object({
  status: literal('ok'),
  version: string(),
  embedding_configured: boolean(),
})

const embeddingConfigSchema = object({
  base_url: string(),
  api_key: string(),
  model: string(),
})

const configSchema = object({
  embedding: union([embeddingConfigSchema, null_()]),
})

const createMemoryResponseSchema = object({
  entry: memoryEntrySchema,
  deduplicated: boolean(),
})

const listMemoriesResponseSchema = object({
  entries: array(memoryEntrySchema),
})

const searchResponseSchema = object({
  results: array(memorySearchResultSchema),
})

const patchMemoryResponseSchema = object({
  entry: memoryEntrySchema,
})

const deleteMemoryResponseSchema = object({
  deleted: string(),
})

const clearMemoriesResponseSchema = object({
  deleted: number(),
})

const putConfigResponseSchema = object({
  embedding: union([embeddingConfigSchema, null_()]),
  backfilled: number(),
})

export type MemoryEntry = InferOutput<typeof memoryEntrySchema>
export type MemorySearchResult = InferOutput<typeof memorySearchResultSchema>
export type MemoryHealth = InferOutput<typeof healthSchema>
export type MemoryEmbeddingConfig = InferOutput<typeof embeddingConfigSchema>
export type MemoryKind = MemoryEntry['kind']

export interface MemoryConfig {
  embedding: MemoryEmbeddingConfig | null
}

export class MemoryServiceError extends Error {}

export class MemoryServiceClient {
  private readonly baseUrl: string
  private readonly fetcher: typeof fetch

  constructor(baseUrl: string, fetcher: typeof fetch = fetch) {
    this.baseUrl = baseUrl.replace(/\/+$/, '')
    this.fetcher = fetcher
  }

  private async request<T>(path: string, parse: ((body: unknown) => T) | undefined, init?: RequestInit): Promise<T> {
    let response: Response
    try {
      response = await this.fetcher(`${this.baseUrl}${path}`, {
        ...init,
        headers: { 'Content-Type': 'application/json', ...init?.headers },
      })
    }
    catch (error) {
      throw new MemoryServiceError(`airi-memory is unreachable at ${this.baseUrl}: ${errorMessageFrom(error) ?? 'unknown network error'}`)
    }

    if (!response.ok) {
      const detail = await response.json().then((body: unknown) =>
        typeof body === 'object' && body !== null && 'detail' in body && typeof body.detail === 'string'
          ? body.detail
          : undefined,
      ).catch(() => undefined)
      throw new MemoryServiceError(detail ?? `airi-memory request failed: ${response.status} ${response.statusText}`)
    }

    if (!parse)
      return undefined as T
    return parse(await response.json())
  }

  async health(): Promise<MemoryHealth> {
    return this.request('/v1/health', body => parse(healthSchema, body))
  }

  async getConfig(): Promise<MemoryConfig> {
    return this.request('/v1/config', body => parse(configSchema, body))
  }

  async putConfig(embedding: MemoryEmbeddingConfig): Promise<MemoryConfig & { backfilled: number }> {
    return this.request('/v1/config', body => parse(putConfigResponseSchema, body), {
      method: 'PUT',
      body: JSON.stringify({ embedding }),
    })
  }

  async createMemory(content: string, sourceSessionId?: string): Promise<{ entry: MemoryEntry, deduplicated: boolean }> {
    return this.request('/v1/memories', body => parse(createMemoryResponseSchema, body), {
      method: 'POST',
      body: JSON.stringify({ content, source_session_id: sourceSessionId }),
    })
  }

  async listMemories(kind?: MemoryKind, q?: string): Promise<MemoryEntry[]> {
    const search = new URLSearchParams()
    if (kind)
      search.set('kind', kind)
    if (q)
      search.set('q', q)
    const query = search.toString()
    return this.request(`/v1/memories${query ? `?${query}` : ''}`, body => parse(listMemoriesResponseSchema, body).entries)
  }

  async patchMemory(id: string, content: string): Promise<MemoryEntry> {
    return this.request(`/v1/memories/${id}`, body => parse(patchMemoryResponseSchema, body).entry, {
      method: 'PATCH',
      body: JSON.stringify({ content }),
    })
  }

  async deleteMemory(id: string): Promise<string> {
    return this.request(`/v1/memories/${id}`, body => parse(deleteMemoryResponseSchema, body).deleted, { method: 'DELETE' })
  }

  async clearMemories(): Promise<number> {
    return this.request('/v1/memories', body => parse(clearMemoriesResponseSchema, body).deleted, { method: 'DELETE' })
  }

  async search(query: string, topK?: number): Promise<MemorySearchResult[]> {
    return this.request('/v1/search', body => parse(searchResponseSchema, body).results, {
      method: 'POST',
      body: JSON.stringify({ query, ...(topK ? { top_k: topK } : {}) }),
    })
  }
}
