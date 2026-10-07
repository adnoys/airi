import type { ChildProcess } from 'node:child_process'

import type { Plugin } from 'vite'

import process from 'node:process'

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

export interface MemorySidecarOptions {
  /** Working directory of the airi-memory service. */
  cwd?: string
  /**
   * Port the service listens on.
   *
   * @default 6430
   */
  port?: number
  /**
   * Health endpoint used for probes.
   *
   * @default '/v1/health'
   */
  healthPath?: string
  /** Explicit interpreter command, for example `/path/to/python`. */
  command?: string
}

const SPAWN_TIMEOUT_MS = 8000
const HEALTH_POLL_INTERVAL_MS = 250

async function isHealthy(port: number, healthPath: string): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 700)
    const response = await fetch(`http://127.0.0.1:${port}${healthPath}`, { signal: controller.signal })
    clearTimeout(timer)
    return response.ok
  }
  catch {
    return false
  }
}

function waitUntilHealthy(port: number, healthPath: string, timeoutMs: number, signal?: AbortSignal): Promise<boolean> {
  return new Promise((resolvePromise) => {
    if (signal?.aborted) {
      resolvePromise(false)
      return
    }

    const startedAt = Date.now()
    let settled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    function finish(ok: boolean) {
      if (settled)
        return
      settled = true
      if (timer)
        clearTimeout(timer)
      resolvePromise(ok)
    }

    signal?.addEventListener('abort', () => finish(false), { once: true })

    const poll = async () => {
      if (settled)
        return
      if (await isHealthy(port, healthPath)) {
        finish(true)
        return
      }
      if (Date.now() - startedAt >= timeoutMs) {
        finish(false)
        return
      }
      timer = setTimeout(poll, HEALTH_POLL_INTERVAL_MS)
    }
    void poll()
  })
}

interface InterpreterCandidate {
  command: string
  prefix: string[]
}

function interpreterCandidates(explicit?: string): InterpreterCandidate[] {
  const candidates: InterpreterCandidate[] = []
  if (explicit)
    candidates.push({ command: explicit, prefix: [] })
  if (process.env.AIRI_MEMORY_PYTHON)
    candidates.push({ command: process.env.AIRI_MEMORY_PYTHON, prefix: [] })
  if (process.platform !== 'win32')
    candidates.push({ command: 'conda', prefix: ['run', '-n', 'airi', '--no-capture-output'] })
  candidates.push({ command: 'python3', prefix: [] }, { command: 'python', prefix: [] })
  return candidates
}

function trySpawn(candidate: InterpreterCandidate, port: number, cwd: string): ChildProcess | undefined {
  const args = [...candidate.prefix, '-m', 'airi_memory', '--port', String(port)]
  try {
    return spawn(candidate.command, args, { cwd, stdio: 'ignore' })
  }
  catch {
    return undefined
  }
}

/**
 * Resolves once the child answers health checks. Aborts early when the spawn
 * fails asynchronously or the process exits before it becomes healthy.
 */
function waitForSpawnedHealth(child: ChildProcess, port: number, healthPath: string): Promise<boolean> {
  return new Promise((resolvePromise) => {
    let settled = false
    const controller = new AbortController()

    function finish(ok: boolean) {
      if (settled)
        return
      settled = true
      controller.abort()
      resolvePromise(ok)
    }

    child.once('error', () => finish(false))
    child.once('exit', () => finish(false))
    void waitUntilHealthy(port, healthPath, SPAWN_TIMEOUT_MS, controller.signal).then(finish)
  })
}

/**
 * Keeps the airi-memory service alive next to the Vite dev server.
 *
 * The application's own memory toggle stays the per-origin switch. This plugin
 * only guarantees that the service the toggle talks to exists while the app is
 * being developed. A service started outside the plugin is reused and never
 * killed.
 */
export function MemorySidecar(options: MemorySidecarOptions = {}): Plugin {
  const portFromEnv = Number(process.env.AIRI_MEMORY_PORT)
  const port = options.port ?? (Number.isFinite(portFromEnv) && portFromEnv > 0 ? portFromEnv : 6430)
  const healthPath = options.healthPath ?? '/v1/health'
  let child: ChildProcess | undefined

  const stopOwnedChild = () => {
    if (!child)
      return
    child.kill()
    child = undefined
  }

  return {
    name: 'airi-memory-sidecar',
    apply: 'serve',
    async configureServer(server) {
      if (process.env.AIRI_MEMORY_SIDECAR === '0')
        return

      const cwd = options.cwd ?? resolve(server.config.root, '..', '..', 'airi-memory')
      if (!existsSync(resolve(cwd, 'airi_memory', '__main__.py'))) {
        server.config.logger.warn(`[airi-memory] service directory not found at ${cwd}, sidecar disabled`)
        return
      }

      if (await isHealthy(port, healthPath)) {
        server.config.logger.info(`[airi-memory] already running on port ${port}, reusing it`)
        return
      }

      for (const candidate of interpreterCandidates(options.command)) {
        const spawned = trySpawn(candidate, port, cwd)
        if (!spawned)
          continue

        const healthy = await waitForSpawnedHealth(spawned, port, healthPath)
        if (healthy) {
          child = spawned
          spawned.once('exit', () => {
            if (child === spawned)
              child = undefined
          })
          server.config.logger.info(`[airi-memory] sidecar started on port ${port} via ${candidate.command}`)
          break
        }

        spawned.kill()
      }

      if (!child) {
        server.config.logger.warn(
          `[airi-memory] could not start the sidecar automatically. Memory stays off until the service is reachable; start it with \`python -m airi_memory\` in ${cwd}`,
        )
        return
      }

      const dispose = () => {
        stopOwnedChild()
        process.off('SIGINT', dispose)
        process.off('SIGTERM', dispose)
      }

      // Vite closes the HTTP server on stop. SIGINT/SIGTERM cover Ctrl+C paths
      // that tear the process down before the close event runs.
      server.httpServer?.once('close', dispose)
      process.once('SIGINT', dispose)
      process.once('SIGTERM', dispose)
    },
  }
}
