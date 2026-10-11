import type { StreamingAssistantMessage } from '../types/chat'

import { nanoid } from 'nanoid'

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

function replySpeech(message: StreamingAssistantMessage): string {
  const sliceText = message.slices
    .filter(slice => slice.type === 'text')
    .map(slice => slice.text)
    .join('')
  if (sliceText.includes(FOLLOW_UP_MESSAGE_MARK))
    return sliceText
  if (typeof message.content === 'string')
    return message.content
  return sliceText
}

/**
 * Stores a marked reply as two assistant messages.
 *
 * The mark may sit in the text slices with no line break after it. Tool results
 * and non-text slices stay on the first message. The follow-up message is speech only.
 */
export function storedAssistantMessages(message: StreamingAssistantMessage): StreamingAssistantMessage[] {
  const parts = splitFollowUpSpeech(replySpeech(message))
  if (!parts)
    return [message]

  const [first, second] = parts
  return [
    {
      ...message,
      content: first,
      slices: [
        { type: 'text', text: first },
        ...message.slices.filter(slice => slice.type !== 'text'),
      ],
      categorization: message.categorization
        ? { reasoning: message.categorization.reasoning, speech: first }
        : undefined,
    },
    {
      role: 'assistant',
      // A missing source id has no stable key. The same literal id makes
      // later follow-up messages collide in one session.
      id: message.id ? `${message.id}:follow` : nanoid(),
      content: second,
      slices: [{ type: 'text', text: second }],
      tool_results: [],
      createdAt: (message.createdAt ?? 0) + 1,
    },
  ]
}
