from __future__ import annotations

import asyncio
import contextlib
import pprint
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from bluesky._vendor.super_state_machine.errors import TransitionError
from bluesky.run_engine import RunEngine
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server.simple import SimpleTiledServer

_ALLOWED_PYTEST_TASKS = {"async_finalizer", "async_setup", "async_teardown"}


# ==================================================================================
# Copied from ophyd-async conftest.py
# ==================================================================================
def _error_and_kill_pending_tasks(
    loop: asyncio.AbstractEventLoop, test_name: str, test_passed: bool
) -> set[asyncio.Task[Any]]:
    """Cancels pending tasks in the event loop for a test. Raises an exception if
    the test hasn't already.

    Args:
        loop: The event loop to check for pending tasks.
        test_name: The name of the test.
        test_passed: Indicates whether the test passed.

    Returns:
        set[asyncio.Task]: The set of unfinished tasks that were cancelled.

    Raises:
        RuntimeError: If there are unfinished tasks and the test didn't fail.
    """
    unfinished_tasks = {
        task
        for task in asyncio.all_tasks(loop)
        if (coro := task.get_coro()) is not None
        and hasattr(coro, "__name__")
        and coro.__name__ not in _ALLOWED_PYTEST_TASKS
        and not task.done()
    }
    for task in unfinished_tasks:
        task.cancel()

    # We only raise an exception here if the test didn't fail anyway.
    # If it did then it makes sense that there's some tasks we need to cancel,
    # but an exception will already have been raised.
    if unfinished_tasks and test_passed:
        msg = (
            f"Not all tasks closed during test {test_name}:\n"
            f"{pprint.pformat(unfinished_tasks, width=88)}"
        )
        raise RuntimeError(msg)

    return unfinished_tasks


@pytest.fixture
def RE(request: pytest.FixtureRequest):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, call_returns_result=True, loop=loop)
    fail_count = request.session.testsfailed

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            with contextlib.suppress(TransitionError):
                RE.halt()

        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()  # type: ignore[reportPrivateUsage]

        try:
            _error_and_kill_pending_tasks(
                loop, request.node.name, request.session.testsfailed == fail_count
            )
        finally:
            loop.close()

    yield RE
    clean_event_loop()


# ==================================================================================


@pytest.fixture
def tiled_client() -> Generator[Container, None, None]:
    server_path = Path("/tmp/pytest/tiled-server")
    if server_path.exists():
        shutil.rmtree(server_path)
    server: SimpleTiledServer = SimpleTiledServer(
        server_path.as_posix(),
        readable_storage=["/tmp/pytest"],  # type: ignore[reportArgumentType]
    )
    client: Container = from_uri(server.uri)
    yield client
    server.close()
