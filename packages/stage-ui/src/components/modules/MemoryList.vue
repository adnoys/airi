<script setup lang="ts">
import type { MemoryKind } from '../../libs/memory/client'

import { errorMessageFrom } from '@moeru/std'
import { Button, FieldInput, IconButton } from '@proj-airi/ui'
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { toast } from 'vue-sonner'

import { useMemoryStore } from '../../stores/modules/memory'

const props = defineProps<{
  /** Show only one kind of memory; omit to list everything. */
  kind?: MemoryKind
}>()

const { t } = useI18n()
const memoryStore = useMemoryStore()

const entries = ref<Awaited<ReturnType<typeof memoryStore.listMemories>>>([])
const filterText = ref('')
const isLoading = ref(false)
const isLoaded = ref(false)

async function refresh() {
  isLoading.value = true
  try {
    entries.value = await memoryStore.listMemories(props.kind, filterText.value || undefined)
    isLoaded.value = true
  }
  catch (error) {
    toast.error(errorMessageFrom(error) ?? t('settings.pages.memory.connection-unreachable'))
  }
  finally {
    isLoading.value = false
  }
}

async function deleteEntry(id: string) {
  try {
    await memoryStore.removeMemory(id)
    entries.value = entries.value.filter(entry => entry.id !== id)
    toast.success(t('settings.pages.memory.memory-deleted'))
  }
  catch (error) {
    toast.error(errorMessageFrom(error) ?? t('settings.pages.memory.connection-unreachable'))
  }
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString()
}

onMounted(refresh)
</script>

<template>
  <div
    :class="[
      'h-fit w-full',
      'flex flex-col gap-3',
      'rounded-xl bg-neutral-100 p-4 dark:bg-[rgba(0,0,0,0.3)]',
    ]"
  >
    <div flex="~ items-center justify-between gap-2">
      <div text-lg font-medium>
        {{ t('settings.pages.memory.list-title') }}
      </div>
      <Button size="sm" :disabled="isLoading" @click="refresh">
        {{ t('settings.pages.memory.refresh') }}
      </Button>
    </div>

    <FieldInput
      v-model="filterText"
      :placeholder="t('settings.pages.memory.search-placeholder')"
      @keydown.enter.prevent="refresh"
    />

    <p v-if="isLoaded && entries.length === 0" text-sm text-neutral-500 dark:text-neutral-400>
      {{ t('settings.pages.memory.list-empty') }}
    </p>

    <ul v-else flex="~ col gap-2">
      <li
        v-for="entry in entries"
        :key="entry.id"
        class="flex items-start justify-between gap-3 rounded-lg bg-neutral-50 p-3 dark:bg-neutral-800/60"
      >
        <div flex="~ col gap-1" min-w-0>
          <p break-words text-sm>
            {{ entry.content }}
          </p>
          <div flex="~ wrap items-center gap-2" text-xs text-neutral-500 dark:text-neutral-400>
            <span
              rounded-full px-2 py-0.5
              :class="entry.kind === 'long'
                ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'
                : 'bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300'"
            >
              {{ t(`settings.pages.memory.kind-${entry.kind}`) }}
            </span>
            <span>{{ t('settings.pages.memory.half-life', { days: Math.round(entry.half_life_days) }) }}</span>
            <span>{{ t('settings.pages.memory.recall-count', { count: entry.recall_count }) }}</span>
            <span>{{ formatDate(entry.created_at) }}</span>
          </div>
        </div>
        <IconButton icon="i-lucide:trash-2" :title="t('settings.pages.memory.delete')" @click="deleteEntry(entry.id)" />
      </li>
    </ul>
  </div>
</template>
