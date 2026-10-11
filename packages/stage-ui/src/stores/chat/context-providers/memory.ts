import type { MemorySearchResult } from '../../../libs/memory/client'
import type { ContextMessage } from '../../../types/chat'

import { ContextUpdateStrategy } from '@proj-airi/server-sdk'
import { nanoid } from 'nanoid'

export const MEMORY_CONTEXT_ID = 'system:memories'

/** Local calendar day of a stored timestamp, matching the machine clock. */
export function memoryLineDate(createdAt: string): string {
  const date = new Date(createdAt)
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  return `${year}-${month}-${day}`
}

const MEMORY_CONTEXT_HEADER = [
  'These lines are background from earlier talks.',
  'Do not answer by acting out a line, quoting it, or explaining these rules.',
  'Do not add events the user did not state.',
].join('\n')

/**
 * Builds the per-request memory context from retrieved results.
 *
 * Like the account context, memories belong to one request and must not be
 * persisted into the context registry: the caller reads freshly retrieved
 * results and passes them in on every model request. Empty results return
 * null, which keeps the prompt free of an empty memory section.
 */
export function createMemoryContext(results: MemorySearchResult[]): ContextMessage | null {
  if (results.length === 0)
    return null

  const lines = results.map((result) => {
    const content = result.content.trim().replace(/\s+/g, ' ')
    return `- [${memoryLineDate(result.created_at)}] ${content}`
  })

  return {
    id: nanoid(),
    contextId: MEMORY_CONTEXT_ID,
    strategy: ContextUpdateStrategy.ReplaceSelf,
    metadata: { source: { id: MEMORY_CONTEXT_ID } },
    text: `${MEMORY_CONTEXT_HEADER}\n\n${lines.join('\n')}`,
    createdAt: Date.now(),
  }
}
