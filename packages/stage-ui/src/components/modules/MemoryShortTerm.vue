<script setup lang="ts">
import { Callout, FieldCheckbox } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { useI18n } from 'vue-i18n'

import MemoryList from './MemoryList.vue'

import { useMemoryStore } from '../../stores/modules/memory'

const { t } = useI18n()
const memoryStore = useMemoryStore()
const { autoExtract } = storeToRefs(memoryStore)
</script>

<template>
  <div flex="~ col gap-4">
    <Callout theme="primary" :label="t('settings.pages.modules.memory-short-term.view-description')" />

    <div
      :class="[
        'h-fit w-full',
        'flex flex-col gap-4',
        'rounded-xl bg-neutral-100 p-4 dark:bg-[rgba(0,0,0,0.3)]',
      ]"
    >
      <FieldCheckbox
        v-model="autoExtract"
        :label="t('settings.pages.memory.auto-extract')"
        :description="t('settings.pages.memory.auto-extract-description')"
      />
    </div>

    <MemoryList kind="short" />
  </div>
</template>
