"""Virtual time system for deterministic fast-forward playback.

This module provides timers that coordinate time across all nodes.
Three execution modes are supported:

1. **Real-time** (speed=1.0): Wall clock time, actual delays
2. **Scaled playback** (speed≠1.0): Faster or slower than real-time
3. **Fast-forward** (speed=inf): Deterministic, as fast as possible

Key components:
- Timer: Protocol that nodes use to access time
- ScaledTimer: Wall clock with optional speed scaling
- FastForwardTimer: Virtual clock driven by playback
- VirtualClock: Coordinator for fast-forward mode
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import time_ns
from typing import Protocol, runtime_checkable


# =============================================================================
# Timer Protocol
# =============================================================================


@runtime_checkable
class Timer(Protocol):
    """Protocol for time access in nodes.

    Nodes request a Timer parameter and the runtime injects the appropriate
    implementation based on execution mode.

    Example:
        >>> async def my_node(
        ...     input: Annotated[In[str], "data"],
        ...     timer: Timer,
        ... ) -> None:
        ...     async for item in input:
        ...         print(f"At {timer.time_ns()}: {item}")
        ...         await timer.sleep(0.1)
    """

    def time_ns(self) -> int:
        """Get current time in nanoseconds."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...

    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until the specified timestamp (no-op if already past)."""
        ...

    def periodic(self, period: float) -> AsyncIterator[int]:
        """Return an async iterator that yields timestamps at regular intervals.

        The first yield happens after one period has elapsed.
        """
        ...


# =============================================================================
# Base Implementation
# =============================================================================


class TimerBase(ABC):
    """Base class for Timer implementations with shared periodic logic."""

    @abstractmethod
    def time_ns(self) -> int:
        """Get current time in nanoseconds."""
        ...

    @abstractmethod
    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified duration."""
        ...

    @abstractmethod
    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until the specified timestamp."""
        ...

    async def periodic(self, period: float) -> AsyncIterator[int]:
        """Yield timestamps at regular intervals.

        Uses sleep_until for precise timing that doesn't drift.
        Each iteration sleeps until the next tick, then yields the current time.

        Args:
            period: Interval between yields in seconds.

        Yields:
            Current timestamp in nanoseconds after each period.

        Example:
            >>> async for t in timer.periodic(1.0):
            ...     print(f"Tick at {t}")
            ...     if should_stop:
            ...         break
        """
        period_ns = int(period * 1_000_000_000)
        next_tick = self.time_ns() + period_ns

        while True:
            await self.sleep_until(next_tick)
            yield self.time_ns()
            next_tick += period_ns


# =============================================================================
# Virtual Clock (coordinator for fast-forward mode)
# =============================================================================


@dataclass(order=True)
class _Waiter:
    """A pending timer wakeup in the virtual clock."""

    wake_time: int
    seq: int = field(compare=True)  # Tiebreaker for stable ordering
    event: asyncio.Event = field(compare=False)


class VirtualClock:
    """Coordinates virtual time across all tasks in fast-forward mode.

    The clock maintains the current simulation time and a heap of pending
    timer wakeups. When advance_to() is called, it wakes all timers whose
    wake time has been reached, allowing them to run before returning.

    This enables deterministic execution: timers at time T fire before
    messages at time T are delivered.

    Invariants:
        - Time only moves forward
        - All waiters with wake_time <= current_time are woken before
          advance_to() returns
        - Woken tasks may schedule new timers; if those are still <= target,
          they are also processed in the same advance_to() call

    Example:
        >>> clock = VirtualClock(start_time=1_000_000_000)
        >>> timer = FastForwardTimer(clock)
        >>>
        >>> # In playback node:
        >>> async for timestamp, msg in messages:
        ...     await clock.advance_to(timestamp)
        ...     publish(msg)
    """

    def __init__(self, start_time: int = 0):
        self._time: int = start_time
        self._waiters: list[_Waiter] = []
        self._seq: int = 0
        self._advancing: bool = False

    def now(self) -> int:
        """Get current virtual time in nanoseconds."""
        return self._time

    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until virtual time reaches target.

        Returns immediately if target is at or before current time.
        Otherwise, blocks until advance_to() reaches the target time.
        """
        if target_ns <= self._time:
            return

        event = asyncio.Event()
        waiter = _Waiter(wake_time=target_ns, seq=self._seq, event=event)
        self._seq += 1
        heappush(self._waiters, waiter)
        await event.wait()

    async def advance_to(self, target_ns: int) -> None:
        """Advance virtual time to target, waking all timers along the way.

        Processing order:
        1. Pop earliest waiter from heap
        2. Set clock to waiter's time
        3. Wake the waiter
        4. Yield control (waiter runs, may schedule more timers)
        5. Repeat until no waiters <= target_ns
        6. Set clock to target_ns

        This ensures that if a woken task schedules another timer that's
        still before target_ns, it will also be processed before returning.

        Args:
            target_ns: Target time to advance to.

        Raises:
            RuntimeError: If called reentrantly (a woken timer tries to
                advance the clock).
        """
        if target_ns < self._time:
            return  # Can't go backwards

        if self._advancing:
            raise RuntimeError(
                "VirtualClock.advance_to() called reentrantly. "
                "A woken timer tried to advance the clock, which is not allowed."
            )

        self._advancing = True
        try:
            # Process all waiters up to target time
            # Note: woken tasks may add new waiters, so we re-check each iteration
            while self._waiters and self._waiters[0].wake_time <= target_ns:
                waiter = heappop(self._waiters)
                self._time = waiter.wake_time
                waiter.event.set()
                # Yield to let the woken task run
                # It may schedule new timers that we need to process
                await asyncio.sleep(0)

            self._time = target_ns
        finally:
            self._advancing = False

    def pending_count(self) -> int:
        """Number of pending timer wakeups."""
        return len(self._waiters)

    def next_wakeup(self) -> int | None:
        """Timestamp of next pending wakeup, or None if no waiters."""
        return self._waiters[0].wake_time if self._waiters else None

    async def flush(self) -> None:
        """Wake all pending timers immediately.

        This is useful at the end of playback to ensure any timers scheduled
        by consumers are fired before the graph shuts down. All waiters are
        woken in timestamp order, with the clock set to each waiter's time.

        After flush completes, pending_count() will be 0.
        """
        if self._advancing:
            raise RuntimeError(
                "VirtualClock.flush() called reentrantly. "
                "A woken timer tried to flush the clock, which is not allowed."
            )

        self._advancing = True
        try:
            while self._waiters:
                waiter = heappop(self._waiters)
                self._time = waiter.wake_time
                waiter.event.set()
                # Yield to let the woken task run (may schedule more timers)
                await asyncio.sleep(0)
                # Yield again to let any tasks spawned by woken task register
                await asyncio.sleep(0)
        finally:
            self._advancing = False


# =============================================================================
# Timer Implementations
# =============================================================================


class ScaledTimer(TimerBase):
    """Timer with optional speed scaling.

    Maintains virtual time that advances at `speed` times real time.
    With speed=1.0, this behaves identically to wall clock time.

    Args:
        speed: Playback speed multiplier.
            - 1.0: real-time (virtual time == wall time)
            - 2.0: 2x speed (sleeps are halved)
            - 0.5: half speed (sleeps are doubled)
        start_time: Initial virtual time in nanoseconds. If None, uses
            current wall clock time.

    Example:
        >>> timer = ScaledTimer(speed=2.0)  # 2x playback
        >>> await timer.sleep(1.0)  # Actually sleeps 0.5 seconds
    """

    def __init__(self, speed: float = 1.0, start_time: int | None = None):
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")
        self._speed = speed
        self._start_real = time_ns()
        self._start_virtual = start_time if start_time is not None else self._start_real

    def time_ns(self) -> int:
        """Get current virtual time based on scaled elapsed real time."""
        elapsed_real = time_ns() - self._start_real
        elapsed_virtual = int(elapsed_real * self._speed)
        return self._start_virtual + elapsed_virtual

    async def sleep(self, seconds: float) -> None:
        """Sleep for the specified virtual duration."""
        if seconds <= 0:
            return
        real_seconds = seconds / self._speed
        await asyncio.sleep(real_seconds)

    async def sleep_until(self, target_ns: int) -> None:
        """Sleep until virtual time reaches target."""
        current = self.time_ns()
        if target_ns > current:
            delay_virtual_ns = target_ns - current
            delay_real_s = delay_virtual_ns / self._speed / 1_000_000_000
            await asyncio.sleep(delay_real_s)


class FastForwardTimer(TimerBase):
    """Timer for fast-forward playback using VirtualClock.

    All operations delegate to the virtual clock. Time only advances
    when the playback node calls clock.advance_to().

    This ensures deterministic execution order: all timers scheduled
    for time T are woken before messages at time T are delivered.

    Args:
        clock: The VirtualClock instance that coordinates time.

    Example:
        >>> clock = VirtualClock()
        >>> timer = FastForwardTimer(clock)
        >>>
        >>> # Timer blocks until clock advances
        >>> await timer.sleep(1.0)
    """

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def time_ns(self) -> int:
        """Get current virtual time from the clock."""
        return self._clock.now()

    async def sleep(self, seconds: float) -> None:
        """Schedule a wakeup and wait for clock to advance."""
        if seconds <= 0:
            return
        delay_ns = int(seconds * 1_000_000_000)
        target = self._clock.now() + delay_ns
        await self._clock.sleep_until(target)

    async def sleep_until(self, target_ns: int) -> None:
        """Wait for clock to advance to target time."""
        await self._clock.sleep_until(target_ns)


# =============================================================================
# Factory
# =============================================================================


def create_timer(
    speed: float = 1.0,
    clock: VirtualClock | None = None,
    start_time: int | None = None,
) -> TimerBase:
    """Create appropriate timer for the execution mode.

    Args:
        speed: Playback speed multiplier.
            - 1.0: real-time
            - 2.0: 2x speed
            - 0.5: half speed
            - float('inf'): fast-forward (requires clock)
        clock: VirtualClock for fast-forward mode. Required if speed=inf.
        start_time: Initial virtual time for ScaledTimer. If None, uses
            current wall clock time.

    Returns:
        Timer instance appropriate for the execution mode.

    Raises:
        ValueError: If speed=inf but no clock provided, or speed <= 0.

    Example:
        >>> # Real-time execution
        >>> timer = create_timer()
        >>>
        >>> # 2x playback
        >>> timer = create_timer(speed=2.0)
        >>>
        >>> # Fast-forward with virtual clock
        >>> clock = VirtualClock()
        >>> timer = create_timer(speed=float('inf'), clock=clock)
    """
    if speed == float("inf"):
        if clock is None:
            raise ValueError("Fast-forward mode (speed=inf) requires a VirtualClock")
        return FastForwardTimer(clock)
    else:
        return ScaledTimer(speed, start_time)
