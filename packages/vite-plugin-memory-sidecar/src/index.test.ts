import type { Server } from 'node:http'

import type { Logger } from 'vite'

import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer as createHttpServer } from 'node:http'
import { createServer as createNetServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createLogger, createServer } from 'vite'
import { afterEach, expect, it } from 'vitest'

import { MemorySidecar } from './index'

const FIXTURE_SERVER = `#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { writeFileSync } from 'node:fs'

const portIndex = process.argv.indexOf('--port')
const port = Number(process.argv[portIndex + 1])
const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000000)'], { stdio: 'ignore' })
writeFileSync(process.env.SIDECAR_PID_FILE, \`\${process.pid}\\n\${child.pid}\\n\`)
createServer((request, response) => {
  response.writeHead(request.url === '/v1/health' ? 200 : 404)
  response.end('ok')
}).listen(port, '127.0.0.1')
`

const originalSidecar = process.env.AIRI_MEMORY_SIDECAR
const originalPython = process.env.AIRI_MEMORY_PYTHON

let directories: string[] = []
let servers: Server[] = []
let viteServers: Array<{ close: () => Promise<void> }> = []
let pids: number[] = []

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createNetServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      const port = typeof address === 'object' && address ? address.port : 0
      server.close(() => resolve(port))
    })
  })
}

function isAlive(pid: number) {
  try {
    process.kill(pid, 0)
    return true
  }
  catch (error) {
    return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EPERM'
  }
}

async function waitUntilDead(pid: number) {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 3000) {
    if (!isAlive(pid))
      return
    await delay(40)
  }
  throw new Error(`process ${pid} is still running`)
}

function captureLogger() {
  const warnings: string[] = []
  const infos: string[] = []
  const logger = createLogger('info', { allowClearScreen: false })
  logger.warn = (message: string) => {
    warnings.push(message)
  }
  logger.info = (message: string) => {
    infos.push(message)
  }
  return { logger: logger as Logger, warnings, infos }
}

async function writeServiceDir(main: string) {
  const root = await mkdtemp(join(tmpdir(), 'airi-memory-sidecar-'))
  directories.push(root)
  await writeFile(join(root, 'index.html'), '<!doctype html><html></html>')
  const packageDir = join(root, 'airi_memory')
  await mkdir(packageDir)
  await writeFile(join(packageDir, '__init__.py'), '')
  await writeFile(join(packageDir, '__main__.py'), main)
  return root
}

async function listen(root: string, port: number, command?: string) {
  const { logger, warnings, infos } = captureLogger()
  const vitePort = await freePort()
  const server = await createServer({
    root,
    configFile: false,
    logLevel: 'info',
    customLogger: logger,
    server: { host: '127.0.0.1', port: vitePort, strictPort: true },
    plugins: [MemorySidecar({ cwd: root, port, command })],
  })
  viteServers.push(server)
  await server.listen()
  return { warnings, infos }
}

afterEach(async () => {
  if (originalSidecar === undefined)
    delete process.env.AIRI_MEMORY_SIDECAR
  else
    process.env.AIRI_MEMORY_SIDECAR = originalSidecar
  if (originalPython === undefined)
    delete process.env.AIRI_MEMORY_PYTHON
  else
    process.env.AIRI_MEMORY_PYTHON = originalPython
  delete process.env.SIDECAR_PID_FILE

  const closing = viteServers
  viteServers = []
  for (const server of closing)
    await server.close()

  const httpServers = servers
  servers = []
  await Promise.all(httpServers.map(server => new Promise<void>((resolve, reject) => {
    server.close(error => error ? reject(error) : resolve())
  })))

  for (const pid of pids) {
    try {
      process.kill(pid, 'SIGKILL')
    }
    catch {
      // The process already exited.
    }
  }
  pids = []

  const roots = directories
  directories = []
  await Promise.all(roots.map(root => rm(root, { recursive: true, force: true })))
})

it('leaves the dev server alone when the sidecar is disabled', async () => {
  process.env.AIRI_MEMORY_SIDECAR = '0'
  delete process.env.AIRI_MEMORY_PYTHON
  const root = await writeServiceDir('')
  const port = await freePort()
  const { infos, warnings } = await listen(root, port, process.execPath)

  expect(infos.join('\n')).not.toContain('sidecar started')
  expect(warnings.join('\n')).not.toContain('could not start')
})

it('warns and skips startup when the service directory is missing', async () => {
  delete process.env.AIRI_MEMORY_SIDECAR
  delete process.env.AIRI_MEMORY_PYTHON
  const root = await mkdtemp(join(tmpdir(), 'airi-memory-sidecar-'))
  directories.push(root)
  await writeFile(join(root, 'index.html'), '<!doctype html><html></html>')
  const port = await freePort()
  const { warnings } = await listen(root, port)

  expect(warnings.join('\n')).toContain(`service directory not found at ${root}`)
})

it('reuses a service that is already healthy and leaves it running', async () => {
  delete process.env.AIRI_MEMORY_SIDECAR
  delete process.env.AIRI_MEMORY_PYTHON
  const port = await freePort()
  const external = createHttpServer((request, response) => {
    response.writeHead(request.url === '/v1/health' ? 200 : 404)
    response.end('ok')
  })
  servers.push(external)
  await new Promise<void>((resolve, reject) => {
    external.once('error', reject)
    external.listen(port, '127.0.0.1', () => resolve())
  })

  const root = await writeServiceDir('raise SystemExit(1)\n')
  const script = join(root, 'fixture.mjs')
  const pidFile = join(root, 'pids')
  await writeFile(script, FIXTURE_SERVER)
  await chmod(script, 0o755)
  process.env.SIDECAR_PID_FILE = pidFile
  const { infos } = await listen(root, port, script)

  expect(infos.join('\n')).toContain(`already running on port ${port}`)
  await expect(readFile(pidFile, 'utf8')).rejects.toThrow()
  await viteServers.pop()?.close()
  const response = await fetch(`http://127.0.0.1:${port}/v1/health`)
  expect(response.status).toBe(200)
})

it('stops the spawned process and the child process it created', async () => {
  delete process.env.AIRI_MEMORY_SIDECAR
  delete process.env.AIRI_MEMORY_PYTHON
  const root = await writeServiceDir('')
  const script = join(root, 'fixture.mjs')
  const pidFile = join(root, 'pids')
  await writeFile(script, FIXTURE_SERVER)
  await chmod(script, 0o755)
  process.env.SIDECAR_PID_FILE = pidFile
  const port = await freePort()
  const { infos } = await listen(root, port, script)

  expect(infos.join('\n')).toContain(`sidecar started on port ${port} via ${script}`)
  const [parentPid, childPid] = (await readFile(pidFile, 'utf8')).trim().split('\n').map(Number)
  if (parentPid === undefined || childPid === undefined)
    throw new Error('pid file is incomplete')
  expect(isAlive(parentPid)).toBe(true)
  expect(isAlive(childPid)).toBe(true)
  pids.push(parentPid, childPid)

  await viteServers.pop()?.close()
  await waitUntilDead(parentPid)
  await waitUntilDead(childPid)
})

it('reports the last interpreter error when startup fails', async () => {
  delete process.env.AIRI_MEMORY_SIDECAR
  delete process.env.AIRI_MEMORY_PYTHON
  const root = await writeServiceDir('import sys\nsys.stderr.write("fixture-start-failed\\n")\nraise SystemExit(1)\n')
  const port = await freePort()
  const { warnings } = await listen(root, port, 'python3')

  expect(warnings.join('\n')).toContain('fixture-start-failed')
}, 30_000)
