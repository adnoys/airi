import type { StreamingAssistantMessage } from '../types/chat'

/** A line the model uses to start the message it sends after the first reply. */
export const FOLLOW_UP_MESSAGE_MARK = '[[follow]]'

/**
 * Splits one reply at the follow-up mark.
 *
 * The first part stays the reply to the user. The second part is the message
 * the character sends next. A missing or empty side keeps the reply whole.
 */
export function splitFollowUpSpeech(text: string): [string, string] | null {
  const index = text.indexOf(FOLLOW_UP_MESSAGE_MARK)
  if (index < 0)
    return null

  const first = text.slice(0, index).trim()
  const second = text.slice(index + FOLLOW_UP_MESSAGE_MARK.length).trim()
  if (first.length === 0 || second.length === 0)
    return null

  return [first, second]
}

/**
 * Stores a marked reply as two assistant messages.
 *
 * Tool results and non-text slices stay on the original message. The follow-up
 * message is speech only.
 */
export function storedAssistantMessages(message: StreamingAssistantMessage): StreamingAssistantMessage[] {
  if (message.tool_results.length > 0)
    return [message]
  if (message.slices.some(slice => slice.type !== 'text'))
    return [message]
  if (typeof message.content !== 'string')
    return [message]

  const parts = splitFollowUpSpeech(message.content)
  if (!parts)
    return [message]

  const [first, second] = parts
  return [
    {
      ...message,
      content: first,
      slices: [{ type: 'text', text: first }],
      categorization: message.categorization
        ? { reasoning: message.categorization.reasoning, speech: first }
        : undefined,
    },
    {
      role: 'assistant',
      id: message.id ? `${message.id}:follow` : 'follow',
      content: second,
      slices: [{ type: 'text', text: second }],
      tool_results: [],
      createdAt: (message.createdAt ?? 0) + 1,
    },
  ]
}
