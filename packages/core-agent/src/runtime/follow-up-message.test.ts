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

  // ROOT CAUSE:
  //
  // Two replies with no id shared one follow-up id.
  // The missing-id branch always wrote 'follow'.
  // A missing source id now gets a new id.
  it('gives each follow-up a distinct id when the source message has none', () => {
    const message = {
      role: 'assistant' as const,
      content: '我爱你。\n[[follow]]\n别走。',
      slices: [{ type: 'text' as const, text: '我爱你。\n[[follow]]\n别走。' }],
      tool_results: [],
    }

    const firstFollowUp = storedAssistantMessages(message)[1]
    const secondFollowUp = storedAssistantMessages(message)[1]

    expect(firstFollowUp?.id).toBeTruthy()
    expect(secondFollowUp?.id).toBeTruthy()
    expect(firstFollowUp?.id).not.toBe(secondFollowUp?.id)
  })

  it('splits when the mark is glued to the next sentence', () => {
    const [first, second] = storedAssistantMessages({
      role: 'assistant',
      id: 'reply-2',
      content: '',
      slices: [{ type: 'text', text: '我爱你。\n\n[[follow]]你听好。' }],
      tool_results: [],
      createdAt: 4,
    })

    expect(first?.content).toBe('我爱你。')
    expect(second?.content).toBe('你听好。')
  })

  it('keeps a sticker on the first message', () => {
    const [first, second] = storedAssistantMessages({
      role: 'assistant',
      id: 'reply-3',
      content: '我爱你。[[follow]]别走。',
      slices: [
        { type: 'text', text: '我爱你。[[follow]]别走。' },
        { type: 'sticker', stickerId: 'heart' },
      ],
      tool_results: [],
    })

    expect(first?.slices).toEqual([
      { type: 'text', text: '我爱你。' },
      { type: 'sticker', stickerId: 'heart' },
    ])
    expect(second?.content).toBe('别走。')
  })
})
