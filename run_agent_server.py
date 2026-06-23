from __future__ import annotations

import argparse

import uvicorn

from app.core.event_loop import configure_asyncio_event_loop_policy


def main() -> None:
    configure_asyncio_event_loop_policy()

    parser = argparse.ArgumentParser(description="Run tongue-agent API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        loop="app.core.event_loop:selector_loop_factory",
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
