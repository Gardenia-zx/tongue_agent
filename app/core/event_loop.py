from __future__ import annotations

import asyncio
import sys


def configure_asyncio_event_loop_policy() -> None:
    if not sys.platform.startswith("win"):
        return

    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:
        return

    asyncio.set_event_loop_policy(selector_policy())


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    configure_asyncio_event_loop_policy()

    if sys.platform.startswith("win"):
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is not None:
            return selector_policy().new_event_loop()

    return asyncio.new_event_loop
