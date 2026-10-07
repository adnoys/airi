<script setup lang="ts">
import type { MemoryRetrievalMode } from '../../libs/memory/client'

import { errorMessageFrom } from '@moeru/std'
import { Button, DoubleCheckButton, FieldCheckbox, FieldInput, FieldSelect } from '@proj-airi/ui'
import { storeToRefs } from 'pinia'
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { toast } from 'vue-sonner'

import MemoryList from './MemoryList.vue'

import { useMemoryStore } from '../../stores/modules/memory'

const { t } = useI18n()
const memoryStore = useMemoryStore()
const { serviceUrl, enabled, autoExtract, connection } = storeToRefs(memoryStore)

const listVersion = ref(0)

const retrievalMode = ref<MemoryRetrievalMode>('embedding')
const embeddingBaseUrl = ref('')
const embeddingApiKey = ref('')
const embeddingModel = ref('')
const llmBaseUrl = ref('')
const llmApiKey = ref('')
const llmModel = ref('')
const isSavingRetrieval = ref(false)
const isCheckingConnection = ref(false)

const newMemoryContent = ref('')
const isAddingMemory = ref(false)

const retrievalModeOptions = computed(() => [
  { label: t('settings.pages.memory.retrieval-embedding'), value: 'embedding' },
  { label: t('settings.pages.memory.retrieval-llm'), value: 'llm' },
])

const connectionLabel = computed(() => {
  if (connection.value === 'connected')
    return t('settings.pages.memory.connection-connected')
  if (connection.value === 'unreachable')
    return t('settings.pages.memory.connection-unreachable')
  return t('settings.pages.memory.connection-unknown')
})

async function checkConnection() {
  isCheckingConnection.value = true
  try {
    const ok = await memoryStore.checkHealth()
    if (ok) {
      toast.success(t('settings.pages.memory.connection-connected'))
      await prefillRetrievalConfig()
    }
    else {
      toast.error(t('settings.pages.memory.connection-unreachable'))
    }
  }
  finally {
    isCheckingConnection.value = false
  }
}

async function prefillRetrievalConfig() {
  try {
    const config = await memoryStore.fetchServiceConfig()
    retrievalMode.value = config.retrieval
    if (config.embedding) {
      embeddingBaseUrl.value = config.embedding.base_url
      embeddingApiKey.value = config.embedding.api_key
      embeddingModel.value = config.embedding.model
    }
    if (config.llm) {
      llmBaseUrl.value = config.llm.base_url
      llmApiKey.value = config.llm.api_key
      llmModel.value = config.llm.model
    }
  }
  catch {
    // The list and health state already signal a broken connection.
  }
}

async function saveRetrievalConfig() {
  const missingFields = (baseUrl: string, model: string) => !baseUrl.trim() || !model.trim()
  if (retrievalMode.value === 'embedding' && missingFields(embeddingBaseUrl.value, embeddingModel.value)) {
    toast.error(t('settings.pages.memory.embedding-missing-fields'))
    return
  }
  if (retrievalMode.value === 'llm' && missingFields(llmBaseUrl.value, llmModel.value)) {
    toast.error(t('settings.pages.memory.embedding-missing-fields'))
    return
  }
  isSavingRetrieval.value = true
  try {
    await memoryStore.pushRetrievalConfig({
      retrieval: retrievalMode.value,
      ...(retrievalMode.value === 'embedding'
        ? { embedding: { base_url: embeddingBaseUrl.value.trim(), api_key: embeddingApiKey.value.trim(), model: embeddingModel.value.trim() } }
        : { llm: { base_url: llmBaseUrl.value.trim(), api_key: llmApiKey.value.trim(), model: llmModel.value.trim() } }),
    })
    toast.success(t('settings.pages.memory.embedding-saved'))
  }
  catch (error) {
    toast.error(errorMessageFrom(error) ?? t('settings.pages.memory.connection-unreachable'))
  }
  finally {
    isSavingRetrieval.value = false
  }
}

async function addMemory() {
  const content = newMemoryContent.value.trim()
  if (!content)
    return
  isAddingMemory.value = true
  try {
    const { deduplicated } = await memoryStore.addMemory(content)
    newMemoryContent.value = ''
    toast.success(deduplicated ? t('settings.pages.memory.memory-duplicate') : t('settings.pages.memory.memory-added'))
  }
  catch (error) {
    toast.error(errorMessageFrom(error) ?? t('settings.pages.memory.connection-unreachable'))
  }
  finally {
    isAddingMemory.value = false
  }
}

async function clearAll() {
  try {
    const count = await memoryStore.clearMemories()
    if (count > 0) {
      toast.success(t('settings.pages.memory.memories-cleared'))
      listVersion.value++
    }
  }
  catch (error) {
    toast.error(errorMessageFrom(error) ?? t('settings.pages.memory.connection-unreachable'))
  }
}

onMounted(async () => {
  if (memoryStore.enabled)
    await checkConnection()
})
</script>

<template>
  <div flex="~ col gap-4">
    <div
      :class="[
        'h-fit w-full',
        'flex flex-col gap-4',
        'rounded-xl bg-neutral-100 p-4 dark:bg-[rgba(0,0,0,0.3)]',
      ]"
    >
      <FieldInput
        v-model="serviceUrl"
        :label="t('settings.pages.memory.service-url')"
        :description="t('settings.pages.memory.service-url-description')"
        placeholder="http://127.0.0.1:6430"
      />

      <div flex="~ wrap items-center gap-2">
        <Button :disabled="isCheckingConnection" @click="checkConnection">
          {{ t('settings.pages.memory.check-connection') }}
        </Button>
        <span text-sm :class="connection === 'connected' ? 'text-lime-600 dark:text-lime-400' : connection === 'unreachable' ? 'text-red-600 dark:text-red-400' : 'text-neutral-500'">
          {{ connectionLabel }}
        </span>
      </div>

      <FieldCheckbox
        v-model="enabled"
        :label="t('settings.pages.memory.enable')"
        :description="t('settings.pages.memory.enable-description')"
      />

      <FieldCheckbox
        v-model="autoExtract"
        :label="t('settings.pages.memory.auto-extract')"
        :description="t('settings.pages.memory.auto-extract-description')"
      />
    </div>

    <div
      :class="[
        'h-fit w-full',
        'flex flex-col gap-4',
        'rounded-xl bg-neutral-100 p-4 dark:bg-[rgba(0,0,0,0.3)]',
      ]"
    >
      <div text-lg font-medium>
        {{ t('settings.pages.memory.embedding-title') }}
      </div>
      <p text-sm text-neutral-500 dark:text-neutral-400>
        {{ t('settings.pages.memory.embedding-description') }}
      </p>

      <FieldSelect
        v-model="retrievalMode"
        :options="retrievalModeOptions"
        :label="t('settings.pages.memory.retrieval-mode')"
        :description="t('settings.pages.memory.retrieval-mode-description')"
      />

      <template v-if="retrievalMode === 'embedding'">
        <FieldInput
          v-model="embeddingBaseUrl"
          :label="t('settings.pages.memory.base-url')"
          :description="t('settings.pages.memory.base-url-description')"
          placeholder="https://api.openai.com/v1"
        />

        <FieldInput
          v-model="embeddingApiKey"
          type="password"
          :label="t('settings.pages.memory.api-key')"
        />

        <FieldInput
          v-model="embeddingModel"
          :label="t('settings.pages.memory.model')"
          :description="t('settings.pages.memory.model-description')"
          placeholder="text-embedding-3-small"
        />
      </template>

      <template v-else>
        <FieldInput
          v-model="llmBaseUrl"
          :label="t('settings.pages.memory.llm-base-url')"
          :description="t('settings.pages.memory.llm-base-url-description')"
          placeholder="https://api.deepseek.com"
        />

        <FieldInput
          v-model="llmApiKey"
          type="password"
          :label="t('settings.pages.memory.api-key')"
        />

        <FieldInput
          v-model="llmModel"
          :label="t('settings.pages.memory.model')"
          :description="t('settings.pages.memory.llm-model-description')"
          placeholder="deepseek-chat"
        />
      </template>

      <Button :disabled="isSavingRetrieval" @click="saveRetrievalConfig">
        {{ t('settings.pages.memory.save-embedding') }}
      </Button>
    </div>

    <div
      :class="[
        'h-fit w-full',
        'flex flex-col gap-4',
        'rounded-xl bg-neutral-100 p-4 dark:bg-[rgba(0,0,0,0.3)]',
      ]"
    >
      <FieldInput
        v-model="newMemoryContent"
        :label="t('settings.pages.memory.add-title')"
        :placeholder="t('settings.pages.memory.add-placeholder')"
        @keydown.enter.prevent="addMemory"
      />
      <Button :disabled="isAddingMemory || !newMemoryContent.trim()" @click="addMemory">
        {{ t('settings.pages.memory.add') }}
      </Button>
    </div>

    <MemoryList :key="listVersion" />

    <div flex="~ justify-end">
      <DoubleCheckButton @confirm="clearAll">
        {{ t('settings.pages.memory.clear-all') }}
      </DoubleCheckButton>
    </div>
  </div>
</template>
