<script setup lang="ts">
import { MemoryShortTerm } from '@proj-airi/stage-ui/components'
</script>

<template>
  <MemoryShortTerm />
</template>

<route lang="yaml">
meta:
  layout: settings
  titleKey: settings.pages.modules.memory-short-term.title
  subtitleKey: settings.title
  stageTransition:
    name: slide
</route>
