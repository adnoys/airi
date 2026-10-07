"""Entry point: ``python -m airi_memory`` from the ``airi-memory`` directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .config import ENV_HOST, ENV_PORT, load_config
from .server import create_app
from .store import MemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description='airi-memory: standalone memory service for AIRI')
    parser.add_argument('--host', default=None, help=f'Default: {ENV_HOST} env or 127.0.0.1')
    parser.add_argument('--port', type=int, default=None, help=f'Default: {ENV_PORT} env or 6430')
    parser.add_argument('--data-dir', type=Path, default=None, help='Where memories.json and config.json live')
    args = parser.parse_args()

    config = load_config(args.data_dir)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.data_dir:
        config.data_dir = args.data_dir

    store = MemoryStore(config.data_dir)
    app = create_app(config, store)

    print(f'airi-memory listening on http://{config.host}:{config.port}')
    print(f'data dir: {config.data_dir}')
    print(f'embedding configured: {config.embedding is not None}')
    if os.environ.get(ENV_HOST) is None and args.host is None:
        print('NOTICE: binding to 127.0.0.1 only; do not expose this service to a network')

    uvicorn.run(app, host=config.host, port=config.port, log_level='info')


if __name__ == '__main__':
    main()
