import { describe, expect, it } from 'vitest'

import { splitFollowUpSpeech, storedAssistantMessages } from './follow-up-message'

describe('splitFollowUpSpeech', () => {
  it('keeps a reply that has no follow-up mark', () => {
    expect(splitFollowUpSpeech('我爱你。')).toBeNull()
  })

  it('splits the repetition from the message the character sends next', () => {
    expect(splitFollowUpSpeech('我爱你。我爱你。\n[[follow]]\n你是我的。')).toEqual([
      '我爱你。我爱你。',
      '你是我的。',
    ])
  })

  it('keeps the reply whole when one side is empty', () => {
    expect(splitFollowUpSpeech('[[follow]]\n你是我的。')).toBeNull()
    expect(splitFollowUpSpeech('我爱你。\n[[follow]]')).toBeNull()
  })
})

describe('storedAssistantMessages', () => {
  it('stores the follow-up as a second assistant message', () => {
    const [first, second] = storedAssistantMessages({
      role: 'assistant',
      id: 'reply-1',
      content: '我爱你。\n[[follow]]\n别走。',
      slices: [{ type: 'text', text: '我爱你。\n[[follow]]\n别走。' }],
      tool_results: [],
      createdAt: 10,
      categorization: { speech: '我爱你。\n[[follow]]\n别走。', reasoning: 'wait' },
    })

    expect(first?.content).toBe('我爱你。')
    expect(first?.slices).toEqual([{ type: 'text', text: '我爱你。' }])
    expect(first?.categorization?.speech).toBe('我爱你。')
    expect(first?.categorization?.reasoning).toBe('wait')
    expect(second?.id).toBe('reply-1:follow')
    expect(second?.content).toBe('别走。')
    expect(second?.createdAt).toBe(11)
  })

  it('does not split a reply that calls a tool', () => {
    const message = {
      role: 'assistant' as const,
      content: '我爱你。\n[[follow]]\n别走。',
      slices: [{ type: 'text' as const, text: '我爱你。\n[[follow]]\n别走。' }],
      tool_results: [{ id: 'call-1', result: 'ok' }],
    }

    expect(storedAssistantMessages(message)).toEqual([message])
  })
})
