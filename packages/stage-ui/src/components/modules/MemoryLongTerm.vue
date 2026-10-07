<script setup lang="ts">
import { Callout } from '@proj-airi/ui'
import { useI18n } from 'vue-i18n'

import MemoryList from './MemoryList.vue'

const { t } = useI18n()
</script>

<template>
  <div flex="~ col gap-4">
    <Callout theme="orange" :label="t('settings.pages.modules.memory-long-term.view-description')" />

    <MemoryList kind="long" />
  </div>
</template>
