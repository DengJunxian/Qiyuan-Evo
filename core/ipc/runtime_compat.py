"""Runtime compatibility helpers for Windows asyncio + pyzmq."""

from __future__ import annotations

import asyncio
import os
import warnings


def ensure_zmq_asyncio_compatibility() -> None:
    """Prefer the selector event loop on Windows for pyzmq asyncio sockets."""
    if os.name != "nt":
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
        if selector_policy is None:
            return
        if isinstance(asyncio.get_event_loop_policy(), selector_policy):
            return
        asyncio.set_event_loop_policy(selector_policy())
