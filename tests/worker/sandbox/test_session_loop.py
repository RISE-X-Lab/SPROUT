"""``SessionLoop``: the event loop that a session drives its sandbox I/O through, from any thread.

An interruption is a ``KeyboardInterrupt`` that a callback on the loop raises while the operation
waits, as a signal handler raises one into ``run_until_complete``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest

from sprout.worker.sandbox.session_loop import SessionLoop


@pytest.fixture
def warnings() -> list[str]:
    return []


@pytest.fixture
def loop(warnings: list[str]) -> Iterator[SessionLoop]:
    """A session loop whose event loop is closed after the test, with the threads of its executor."""
    session_loop = SessionLoop(warn=warnings.append)
    yield session_loop
    event_loop = session_loop.get()
    event_loop.run_until_complete(event_loop.shutdown_default_executor())
    event_loop.close()


def interrupt() -> None:
    raise KeyboardInterrupt("interrupted")


def test_an_interrupted_operation_is_cancelled_in_its_own_task_and_the_loop_serves_the_next_call(
    loop, warnings
):
    """A cancel scope of anyio, as httpx opens one, must be left in the task that entered it."""
    tasks = []

    async def operation() -> None:
        with anyio.fail_after(60):
            tasks.append(asyncio.current_task())
            asyncio.get_running_loop().call_soon(interrupt)
            try:
                await asyncio.Event().wait()
            finally:
                tasks.append(asyncio.current_task())

    with pytest.raises(KeyboardInterrupt, match="^interrupted$"):
        loop.drive(operation())
    after = loop.drive(asyncio.sleep(0, result="next"))

    assert (len(tasks), tasks[0] is tasks[1], after, warnings) == (2, True, "next", [])
    assert asyncio.all_tasks(loop.get()) == set()


def test_an_operation_that_does_not_settle_in_its_grace_period_quarantines_the_loop_until_it_settles(
    loop, warnings
):
    """Until then, a later operation never starts: the sandbox may still be busy with the earlier one."""
    loop.cancel_grace_seconds = 0.01
    release = asyncio.Event()
    started = []

    async def stubborn() -> None:
        asyncio.get_running_loop().call_soon(interrupt)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()

    async def later() -> None:
        started.append("later")

    with pytest.raises(KeyboardInterrupt):
        loop.drive(stubborn())
    with pytest.raises(
        RuntimeError, match="^Previous sandbox operation has not settled; session is quarantined$"
    ):
        loop.drive(later())
    release.set()
    loop.get().run_until_complete(asyncio.wait_for(loop.unsettled, timeout=10))
    after = loop.drive(asyncio.sleep(0, result="next"))

    assert (started, after) == ([], "next")
    assert warnings == ["Interrupted sandbox operation did not settle; quarantining session"]


def test_callers_on_two_threads_take_turns_on_the_loop(loop):
    """The second caller asks while the first operation runs. A second entry into a running loop raises."""
    first_running, second_asked, second_ran = threading.Event(), threading.Event(), threading.Event()
    order = []

    async def first() -> str:
        order.append("first starts")
        first_running.set()
        await asyncio.to_thread(second_asked.wait, 10)
        # Set only if the second operation runs while this one does.
        await asyncio.to_thread(second_ran.wait, 0.2)
        order.append("first ends")
        return "first"

    async def second() -> str:
        second_ran.set()
        order.append("second runs")
        return "second"

    with ThreadPoolExecutor(max_workers=2) as threads:
        one = threads.submit(loop.drive, first())
        first_running.wait(10)
        two = threads.submit(loop.drive, second())
        second_asked.set()
        results = (one.result(10), two.result(10))

    assert (results, order) == (("first", "second"), ["first starts", "first ends", "second runs"])
