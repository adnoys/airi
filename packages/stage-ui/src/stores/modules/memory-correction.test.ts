import { describe, expect, it } from 'vitest'

import { reviseMemoryFromDialogue } from './memory-correction'

describe('reviseMemoryFromDialogue', () => {
  it('deletes a user memory the user lines do not contain', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'yandere', content: '用户昨天问过病娇是什么。' },
      { userText: '我想被在乎', assistantText: '病娇就是占有欲很强' },
    )

    expect(revision).toEqual({ id: 'yandere', action: 'delete' })
  })

  it('replaces a memory when the user line negates that phrase', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'cats', content: '用户喜欢猫' },
      { userText: '我不喜欢猫', assistantText: '' },
    )

    expect(revision).toEqual({ id: 'cats', action: 'replace', content: '用户原话：我不喜欢猫' })
  })

  it('keeps a user memory when the user line contains it', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'tea', content: '用户喜欢抹茶' },
      { userText: '我喜欢抹茶', assistantText: '' },
    )

    expect(revision).toBeNull()
  })

  it('replaces a mixed memory with the user line that matches', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'mixed', content: '用户问过病娇，也喜欢抹茶' },
      { userText: '我喜欢抹茶', assistantText: '病娇是一种占有' },
    )

    expect(revision).toEqual({ id: 'mixed', action: 'replace', content: '用户原话：我喜欢抹茶' })
  })

  it('keeps an assistant memory the assistant line contains', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'name', content: 'AI助手自称 adnoys' },
      { userText: '', assistantText: '我是 adnoys' },
    )

    expect(revision).toBeNull()
  })

  it('deletes an assistant memory the assistant line does not contain', () => {
    const revision = reviseMemoryFromDialogue(
      { id: 'role', content: 'AI助手自称病娇' },
      { userText: '你好', assistantText: '我是 adnoys' },
    )

    expect(revision).toEqual({ id: 'role', action: 'delete' })
  })
})
