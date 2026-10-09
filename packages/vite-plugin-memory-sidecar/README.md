# @proj-airi/vite-plugin-memory-sidecar

A Vite plugin that keeps the local [airi-memory](../../../airi-memory/README.md) service running next to a Vite dev server, so memory works as soon as the application opens instead of requiring a separately started process.

## What it does

- On dev server start, first probes the service health endpoint. When something already listens there (for example an instance the user started by hand), the plugin leaves it alone.
- Otherwise it spawns the Python service as a child process. The first candidate is `conda run -n airi python -m airi_memory`. The `python` token is required. `conda run` treats the next token as the command. Later candidates are `python3` and `python`.
- On dev server stop, the plugin stops the spawned process and every child process it created. An externally started service is never killed.

The spawned service binds to `127.0.0.1` only and stores data in `airi-memory/data/`, which is gitignored. The in-app memory toggle stays the single per-application switch that decides whether chat talks to the service.

## Usage

```ts
import { MemorySidecar } from '@proj-airi/vite-plugin-memory-sidecar'

export default defineConfig({
  plugins: [
    MemorySidecar({ cwd: resolve(import.meta.dirname, '..', '..', 'airi-memory') }),
  ],
})
```

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `cwd` | repo-root `airi-memory` | Working directory of the service |
| `port` | `6430` | Health-check and spawn port |
| `healthPath` | `/v1/health` | Health endpoint used for probes |
| `command` | env or candidates | Explicit interpreter command to use |

Environment:

- `AIRI_MEMORY_SIDECAR=0` disables the plugin at runtime
- `AIRI_MEMORY_PYTHON` names the interpreter to prefer
- `AIRI_MEMORY_PORT` overrides the default listen port when `port` is not set

## Tests

Run `pnpm -F @proj-airi/vite-plugin-memory-sidecar test`.

## When not to use it

- Do not add this to production builds (`apply: 'serve'` keeps it dev-only). Hosted deployments need a properly managed service, not a dev-server child process.
