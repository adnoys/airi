import type { Message } from '@xsai/shared-chat'

import type { MemoryConfig, MemoryEntry, MemoryKind, MemoryRetrievalConfigPayload, MemorySearchResult } from '../../libs/memory/client'

import { errorMessageFrom } from '@moeru/std'
import { chatMessagesToTurns, streamFrom } from '@proj-airi/core-agent'
import { useLocalStorageManualReset } from '@proj-airi/stage-shared/composables'
import { defineStore } from 'pinia'
import { computed, ref, shallowRef } from 'vue'

import { MemoryServiceClient } from '../../libs/memory/client'
import { useConsciousnessStore } from './consciousness'
import { reviseMemoryFromDialogue } from './memory-correction'

export const DEFAULT_MEMORY_SERVICE_URL = 'http://127.0.0.1:6430'

const MAX_EXTRACTED_MEMORIES = 5

export const MEMORY_EXTRACTION_SYSTEM_PROMPT = `You extract lasting facts from one conversation turn. Reply with a JSON array of at most ${MAX_EXTRACTED_MEMORIES} short strings. Each string is one third-person fact about the user or the companion: a preference, a personal detail, a promise, or an event that already happened. Write it as history, not as an order for later replies. Do not store a request to repeat a phrase or to perform a tone as a standing task. Store only what the user stated. Skip details that appear only in the assistant reply. Skip small talk, momentary emotions, and anything temporary. If the user denies a past statement, store nothing about that denial. Reply in the language of the conversation. Output only the JSON array.`

function buildExtractionInput(userText: string, assistantText: string) {
  return `USER SAID:\n${userText}\n\nASSISTANT REPLIED:\n${assistantText}`
}

/** Tolerates markdown fences and stray prose around the JSON array. */
export function parseExtractionResponse(raw: string): string[] {
  const withoutFences = raw.trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/, '')
  const start = withoutFences.indexOf('[')
  const end = withoutFences.lastIndexOf(']')
  if (start < 0 || end <= start)
    return []
  try {
    const parsed: unknown = JSON.parse(withoutFences.slice(start, end + 1))
    if (!Array.isArray(parsed))
      return []
    return parsed
      .filter((item): item is string => typeof item === 'string')
      .map(item => item.trim())
      .filter(item => item.length > 0)
      .slice(0, MAX_EXTRACTED_MEMORIES)
  }
  catch {
    return []
  }
}

/**
 * Client-side facade of the airi-memory service (a separate Python process).
 *
 * The service owns the data; this store owns the AIRI-side settings, the
 * per-request retrieval cache that the chat snapshot reads, and the
 * post-turn extraction that feeds new memories back through the protocol.
 */
export const useMemoryStore = defineStore('memory', () => {
  const consciousnessStore = useConsciousnessStore()

  const serviceUrl = useLocalStorageManualReset<string>('settings/memory/service-url', DEFAULT_MEMORY_SERVICE_URL)
  const enabled = useLocalStorageManualReset<boolean>('settings/memory/enabled', false)
  const autoExtract = useLocalStorageManualReset<boolean>('settings/memory/auto-extract', false)

  const connection = ref<'unknown' | 'connected' | 'unreachable'>('unknown')
  const isExtracting = ref(false)

  /** Retrieval result for the in-flight model request; never persisted across requests. */
  const retrievedMemories = shallowRef<MemorySearchResult[]>([])

  const configured = computed(() => enabled.value && serviceUrl.value.trim().length > 0)

  function getClient(): MemoryServiceClient {
    return new MemoryServiceClient(serviceUrl.value)
  }

  async function checkHealth(): Promise<boolean> {
    try {
      await getClient().health()
      connection.value = 'connected'
      return true
    }
    catch {
      connection.value = 'unreachable'
      return false
    }
  }

  /**
   * Retrieves memories for one user message and caches them for the request
   * snapshot. Chat calls this after the user message is already visible.
   * A failing service degrades to no memories and must not fail the turn.
   */
  async function retrieveForQuery(query: string): Promise<void> {
    if (!configured.value) {
      retrievedMemories.value = []
      return
    }
    try {
      retrievedMemories.value = await getClient().search(query)
      connection.value = 'connected'
    }
    catch {
      retrievedMemories.value = []
      connection.value = 'unreachable'
    }
  }

  /**
   * Extracts lasting facts from a finished turn with the chat model and
   * stores them through the service. Fire-and-forget by design: callers
   * must not await it on the reply path.
   */
  async function extractFromTurn(userText: string, assistantText: string, sessionId?: string): Promise<number> {
    if (!configured.value || !autoExtract.value || isExtracting.value)
      return 0

    const providerId = consciousnessStore.activeProvider
    const modelId = consciousnessStore.activeModel
    if (!providerId || !modelId)
      return 0

    isExtracting.value = true
    try {
      const chatProvider = await consciousnessStore.getChatProviderInstance(providerId)
      const messages: Message[] = [
        { role: 'system', content: MEMORY_EXTRACTION_SYSTEM_PROMPT },
        { role: 'user', content: buildExtractionInput(userText, assistantText) },
      ]
      let responseText = ''
      await streamFrom({
        model: modelId,
        chatProvider,
        conversation: { turns: chatMessagesToTurns(messages) },
        options: {
          onStreamEvent: (event) => {
            if (event.type === 'text-delta')
              responseText += event.text
          },
        },
      })

      const client = getClient()
      let stored = 0
      for (const candidate of parseExtractionResponse(responseText)) {
        try {
          await client.createMemory(candidate, sessionId)
          stored++
        }
        catch (error) {
          console.warn('[memory] Failed to store an extracted memory:', errorMessageFrom(error))
        }
      }
      connection.value = 'connected'
      return stored
    }
    catch (error) {
      console.warn('[memory] Extraction failed:', errorMessageFrom(error))
      connection.value = 'unreachable'
      return 0
    }
    finally {
      isExtracting.value = false
    }
  }

  async function pushRetrievalConfig(payload: MemoryRetrievalConfigPayload): Promise<MemoryConfig & { backfilled: number }> {
    return await getClient().putConfig(payload)
  }

  async function fetchServiceConfig(): Promise<MemoryConfig> {
    return await getClient().getConfig()
  }

  async function listMemories(kind?: MemoryKind, q?: string): Promise<MemoryEntry[]> {
    return await getClient().listMemories(kind, q)
  }

  async function addMemory(content: string): Promise<{ entry: MemoryEntry, deduplicated: boolean }> {
    return await getClient().createMemory(content)
  }

  async function updateMemory(id: string, content: string): Promise<MemoryEntry> {
    return await getClient().patchMemory(id, content)
  }

  async function removeMemory(id: string): Promise<void> {
    await getClient().deleteMemory(id)
  }

  /**
   * Rewrites retrieved memories from the dialogues that created them.
   *
   * Chat calls this after retrieval and before the model reads
   * `retrievedMemories`. Each memory is checked against its source session.
   * Memories from that same session are checked too. A missing dialogue is
   * left unchanged. A failing service must not fail the turn.
   */
  async function reconcileRetrievedMemories(
    loadDialogue: (sessionId: string) => Promise<{ userText: string, assistantText: string } | null>,
  ): Promise<number> {
    if (!configured.value || retrievedMemories.value.length === 0)
      return 0

    const sessionIds = new Set(
      retrievedMemories.value.flatMap(memory => memory.source_session_id ? [memory.source_session_id] : []),
    )
    if (sessionIds.size === 0)
      return 0

    try {
      const listed = await getClient().listMemories()
      const candidates = new Map<string, { id: string, content: string, source_session_id: string }>()
      for (const memory of [...retrievedMemories.value, ...listed]) {
        if (!memory.source_session_id || !sessionIds.has(memory.source_session_id))
          continue
        candidates.set(memory.id, {
          id: memory.id,
          content: memory.content,
          source_session_id: memory.source_session_id,
        })
      }

      const dialogues = new Map<string, { userText: string, assistantText: string } | null>()
      const applied = new Map<string, { action: 'delete' } | { action: 'replace', content: string }>()
      for (const memory of candidates.values()) {
        if (!dialogues.has(memory.source_session_id))
          dialogues.set(memory.source_session_id, await loadDialogue(memory.source_session_id))
        const dialogue = dialogues.get(memory.source_session_id)
        if (!dialogue)
          continue
        const revision = reviseMemoryFromDialogue(memory, dialogue)
        if (!revision)
          continue
        try {
          if (revision.action === 'delete')
            await getClient().deleteMemory(memory.id)
          else
            await getClient().patchMemory(memory.id, revision.content)
          applied.set(memory.id, revision)
        }
        catch (error) {
          console.warn('[memory] Failed to revise a memory from its source dialogue:', errorMessageFrom(error))
        }
      }

      if (applied.size > 0) {
        retrievedMemories.value = retrievedMemories.value.flatMap((memory) => {
          const revision = applied.get(memory.id)
          if (!revision)
            return [memory]
          if (revision.action === 'delete')
            return []
          return [{ ...memory, content: revision.content }]
        })
      }
      connection.value = 'connected'
      return applied.size
    }
    catch (error) {
      console.warn('[memory] Failed to compare memories with their source dialogues:', errorMessageFrom(error))
      connection.value = 'unreachable'
      return 0
    }
  }

  async function clearMemories(): Promise<number> {
    return await getClient().clearMemories()
  }

  function resetState() {
    enabled.reset()
    autoExtract.reset()
    serviceUrl.reset()
    retrievedMemories.value = []
  }

  return {
    serviceUrl,
    enabled,
    autoExtract,
    connection,
    isExtracting,
    retrievedMemories,
    configured,
    getClient,
    checkHealth,
    retrieveForQuery,
    extractFromTurn,
    pushRetrievalConfig,
    fetchServiceConfig,
    listMemories,
    addMemory,
    updateMemory,
    removeMemory,
    reconcileRetrievedMemories,
    clearMemories,
    resetState,
  }
})
