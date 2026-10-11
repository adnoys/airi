/**
 * Words that name the speaker or the act of remembering.
 * They are not evidence that a stored event happened.
 */
const TERM_STOPWORDS = [
  '是什么',
  '什么',
  '这个',
  '那个',
  '一个',
  '用户',
  '助手',
  '原话',
  '昨天',
  '今天',
  '说过',
  '问过',
  '问了',
  '没有',
  '从来',
  '不是',
  '要求',
  '了解',
  '想要',
  '自己',
  '自称',
  '明确',
  '表示',
  '已经',
  '可以',
  '就是',
  '知道',
  '觉得',
  '让我',
  '对其',
  '看到',
  '讨论过',
  '也',
  '还',
  '并',
  '和',
]

const STOPWORDS_BY_LENGTH = [...TERM_STOPWORDS].sort((left, right) => right.length - left.length)

export interface SourceDialogue {
  userText: string
  assistantText: string
}

export type MemoryRevision
  = | { id: string, action: 'delete' }
    | { id: string, action: 'replace', content: string }

function memorySide(content: string): 'user' | 'assistant' {
  if (/^(?:AI|助手)/.test(content.trim()))
    return 'assistant'
  return 'user'
}

function salientParts(text: string): string[] {
  const parts: string[] = []
  const chunks = text.match(/[\u4E00-\u9FFF]+/g)
  if (chunks) {
    for (const chunk of chunks) {
      let pending = [chunk]
      for (const stopword of STOPWORDS_BY_LENGTH) {
        const next: string[] = []
        for (const piece of pending) {
          for (const part of piece.split(stopword)) {
            if (part.length >= 2)
              next.push(part)
          }
        }
        pending = next
      }
      parts.push(...pending)
    }
  }
  const latin = text.match(/[A-Z][\w-]{2,}/gi)
  if (latin)
    parts.push(...latin)
  return parts
}

/**
 * A phrase inside a denial is not evidence for the stored claim.
 * 「我不喜欢猫」 contains 「喜欢猫」 and still denies it.
 */
function phraseSupport(part: string, side: string): 'affirmed' | 'negated' | 'absent' {
  let negated = false
  let from = 0
  while (from < side.length) {
    const at = side.indexOf(part, from)
    if (at < 0)
      break
    const previous = side.slice(Math.max(0, at - 2), at)
    const denied = /[不没未]$/.test(previous) || previous === '不是' || previous === '没有' || previous === '并未'
    if (!denied)
      return 'affirmed'
    negated = true
    from = at + part.length
  }
  return negated ? 'negated' : 'absent'
}

/** The whole line that contains the claim. A clipped window would change the meaning. */
function quoteFrom(side: string, parts: readonly string[]): string | null {
  let best: string | null = null
  let bestScore = 0
  for (const line of side.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed)
      continue
    let score = 0
    for (const part of parts) {
      if (trimmed.includes(part))
        score += part.length
    }
    if (score > bestScore) {
      best = trimmed
      bestScore = score
    }
  }
  return best
}

/**
 * Revises one memory from the dialogue that produced it.
 *
 * Returns null when the dialogue affirms the memory as a whole phrase. A
 * user memory is compared with user lines. An assistant memory is compared
 * with assistant lines. No overlap deletes the memory. A denial, or a mix of
 * phrases, replaces the memory with the whole original line.
 */
export function reviseMemoryFromDialogue(
  memory: { id: string, content: string },
  dialogue: SourceDialogue,
): MemoryRevision | null {
  const parts = salientParts(memory.content)
  if (parts.length === 0)
    return null

  const sideName = memorySide(memory.content)
  const side = sideName === 'assistant' ? dialogue.assistantText : dialogue.userText
  const supports = parts.map(part => phraseSupport(part, side))
  if (supports.every(support => support === 'affirmed'))
    return null
  if (supports.every(support => support === 'absent'))
    return { id: memory.id, action: 'delete' }

  const quote = quoteFrom(side, parts)
  if (!quote)
    return { id: memory.id, action: 'delete' }
  const content = `${sideName === 'assistant' ? '助手原话' : '用户原话'}：${quote}`
  if (content === memory.content)
    return null
  return { id: memory.id, action: 'replace', content }
}
