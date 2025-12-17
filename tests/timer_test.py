"""Tests for the timer module."""

import asyncio
from time import time_ns

import pytest

from tinman.timer import (
    FastForwardTimer,
    ScaledTimer,
    Timer,
    VirtualClock,
    create_timer,
)


# =============================================================================
# VirtualClock Tests
# =============================================================================


class TestVirtualClock:
    """Tests for VirtualClock."""

    def test_initial_time(self) -> None:
        """Clock starts at specified time."""
        clock = VirtualClock(start_time=1000)
        assert clock.now() == 1000

    def test_initial_time_default(self) -> None:
        """Clock defaults to time 0."""
        clock = VirtualClock()
        assert clock.now() == 0

    @pytest.mark.asyncio
    async def test_advance_to_updates_time(self) -> None:
        """advance_to updates the current time."""
        clock = VirtualClock(start_time=100)
        await clock.advance_to(500)
        assert clock.now() == 500

    @pytest.mark.asyncio
    async def test_advance_to_backwards_is_noop(self) -> None:
        """advance_to with earlier time does nothing."""
        clock = VirtualClock(start_time=500)
        await clock.advance_to(100)
        assert clock.now() == 500

    @pytest.mark.asyncio
    async def test_sleep_until_past_returns_immediately(self) -> None:
        """sleep_until with past time returns immediately."""
        clock = VirtualClock(start_time=500)
        # Should not block
        await asyncio.wait_for(clock.sleep_until(100), timeout=0.1)

    @pytest.mark.asyncio
    async def test_sleep_until_present_returns_immediately(self) -> None:
        """sleep_until with current time returns immediately."""
        clock = VirtualClock(start_time=500)
        await asyncio.wait_for(clock.sleep_until(500), timeout=0.1)

    @pytest.mark.asyncio
    async def test_sleep_until_future_blocks(self) -> None:
        """sleep_until with future time blocks until advance."""
        clock = VirtualClock(start_time=100)
        woken = False

        async def sleeper() -> None:
            nonlocal woken
            await clock.sleep_until(500)
            woken = True

        task = asyncio.create_task(sleeper())
        await asyncio.sleep(0)  # Let sleeper start
        assert not woken

        await clock.advance_to(500)
        await asyncio.sleep(0)  # Let sleeper complete
        assert woken
        await task

    @pytest.mark.asyncio
    async def test_advance_wakes_multiple_waiters(self) -> None:
        """advance_to wakes all waiters up to target time."""
        clock = VirtualClock(start_time=0)
        wake_times: list[int] = []

        async def sleeper(target: int) -> None:
            await clock.sleep_until(target)
            wake_times.append(clock.now())

        tasks = [
            asyncio.create_task(sleeper(100)),
            asyncio.create_task(sleeper(200)),
            asyncio.create_task(sleeper(300)),
        ]
        await asyncio.sleep(0)  # Let all sleepers start

        await clock.advance_to(250)
        await asyncio.gather(*tasks[:2])  # First two should complete

        assert wake_times == [100, 200]
        assert clock.now() == 250

        # Third waiter still pending
        assert clock.pending_count() == 1
        assert clock.next_wakeup() == 300

        await clock.advance_to(300)
        await tasks[2]
        assert wake_times == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_advance_processes_in_order(self) -> None:
        """Waiters are woken in timestamp order."""
        clock = VirtualClock(start_time=0)
        wake_order: list[str] = []

        async def sleeper(name: str, target: int) -> None:
            await clock.sleep_until(target)
            wake_order.append(name)

        # Schedule out of order
        tasks = [
            asyncio.create_task(sleeper("c", 300)),
            asyncio.create_task(sleeper("a", 100)),
            asyncio.create_task(sleeper("b", 200)),
        ]
        await asyncio.sleep(0)

        await clock.advance_to(300)
        await asyncio.gather(*tasks)

        assert wake_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_woken_task_can_schedule_more_timers(self) -> None:
        """A woken task can schedule additional timers that get processed."""
        clock = VirtualClock(start_time=0)
        events: list[str] = []

        async def first_sleeper() -> None:
            await clock.sleep_until(100)
            events.append("first@100")
            # Schedule another timer at 150 (still before advance target of 200)
            asyncio.create_task(second_sleeper())

        async def second_sleeper() -> None:
            await clock.sleep_until(150)
            events.append("second@150")

        task1 = asyncio.create_task(first_sleeper())
        await asyncio.sleep(0)

        # Advance to 200 - should process both 100 and 150
        await clock.advance_to(200)
        await task1
        await asyncio.sleep(0)  # Let second task complete

        assert events == ["first@100", "second@150"]
        assert clock.now() == 200

    @pytest.mark.asyncio
    async def test_reentrant_advance_raises(self) -> None:
        """advance_to raises if called reentrantly."""
        clock = VirtualClock(start_time=0)
        error_raised = False

        async def bad_sleeper() -> None:
            nonlocal error_raised
            await clock.sleep_until(100)
            # This should raise - can't advance from within a woken callback
            try:
                await clock.advance_to(200)
            except RuntimeError as e:
                if "reentrantly" in str(e):
                    error_raised = True
                raise

        task = asyncio.create_task(bad_sleeper())
        await asyncio.sleep(0)

        # The advance_to itself completes, but the task gets the error
        await clock.advance_to(100)
        await asyncio.sleep(0)  # Let task handle the error

        assert error_raised, "Expected RuntimeError for reentrant advance"

        # Clean up the task (it raised an exception)
        try:
            await task
        except RuntimeError:
            pass

    def test_pending_count(self) -> None:
        """pending_count returns number of waiters."""
        clock = VirtualClock()
        assert clock.pending_count() == 0

    def test_next_wakeup_empty(self) -> None:
        """next_wakeup returns None when no waiters."""
        clock = VirtualClock()
        assert clock.next_wakeup() is None

    @pytest.mark.asyncio
    async def test_flush_wakes_all_timers(self) -> None:
        """flush() wakes all pending timers in order."""
        clock = VirtualClock(start_time=0)
        wake_order: list[int] = []

        async def sleeper(target: int) -> None:
            await clock.sleep_until(target)
            wake_order.append(target)

        tasks = [
            asyncio.create_task(sleeper(300)),
            asyncio.create_task(sleeper(100)),
            asyncio.create_task(sleeper(200)),
        ]
        await asyncio.sleep(0)  # Let sleepers start

        assert clock.pending_count() == 3
        await clock.flush()
        await asyncio.gather(*tasks)

        assert wake_order == [100, 200, 300]
        assert clock.pending_count() == 0
        assert clock.now() == 300  # Clock advanced to last timer

    @pytest.mark.asyncio
    async def test_flush_handles_cascading_timers(self) -> None:
        """flush() handles timers that schedule more timers."""
        clock = VirtualClock(start_time=0)
        events: list[str] = []

        async def cascading_sleeper() -> None:
            await clock.sleep_until(100)
            events.append("first@100")
            # Schedule another timer
            asyncio.create_task(second_sleeper())

        async def second_sleeper() -> None:
            await clock.sleep_until(200)
            events.append("second@200")

        task = asyncio.create_task(cascading_sleeper())
        await asyncio.sleep(0)

        await clock.flush()
        await task
        await asyncio.sleep(0)  # Let second task complete

        assert events == ["first@100", "second@200"]

    @pytest.mark.asyncio
    async def test_flush_empty_clock(self) -> None:
        """flush() on empty clock does nothing."""
        clock = VirtualClock(start_time=100)
        await clock.flush()  # Should not raise
        assert clock.now() == 100


# =============================================================================
# ScaledTimer Tests
# =============================================================================


class TestScaledTimer:
    """Tests for ScaledTimer."""

    def test_time_ns_realtime(self) -> None:
        """With speed=1.0, time_ns tracks wall clock."""
        timer = ScaledTimer(speed=1.0)
        before = time_ns()
        t = timer.time_ns()
        after = time_ns()
        assert before <= t <= after

    def test_time_ns_with_start_time(self) -> None:
        """start_time offsets the virtual time."""
        start = 1_000_000_000_000  # 1000 seconds in the future
        timer = ScaledTimer(speed=1.0, start_time=start)
        t = timer.time_ns()
        # Should be close to start_time (within a few ms)
        assert abs(t - start) < 10_000_000  # 10ms tolerance

    def test_invalid_speed_raises(self) -> None:
        """Speed <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ScaledTimer(speed=0)
        with pytest.raises(ValueError, match="positive"):
            ScaledTimer(speed=-1.0)

    @pytest.mark.asyncio
    async def test_sleep_zero_returns_immediately(self) -> None:
        """sleep(0) returns immediately."""
        timer = ScaledTimer()
        await asyncio.wait_for(timer.sleep(0), timeout=0.1)

    @pytest.mark.asyncio
    async def test_sleep_negative_returns_immediately(self) -> None:
        """sleep with negative value returns immediately."""
        timer = ScaledTimer()
        await asyncio.wait_for(timer.sleep(-1.0), timeout=0.1)

    @pytest.mark.asyncio
    async def test_sleep_scaled(self) -> None:
        """sleep duration is scaled by speed."""
        timer = ScaledTimer(speed=10.0)  # 10x speed
        start = time_ns()
        await timer.sleep(0.1)  # Should sleep ~10ms real time
        elapsed = time_ns() - start
        # Allow generous tolerance for CI variability
        assert elapsed < 50_000_000  # Less than 50ms

    @pytest.mark.asyncio
    async def test_sleep_until_past(self) -> None:
        """sleep_until with past time returns immediately."""
        timer = ScaledTimer(speed=1.0, start_time=1000)
        await asyncio.wait_for(timer.sleep_until(500), timeout=0.1)

    def test_conforms_to_protocol(self) -> None:
        """ScaledTimer conforms to Timer protocol."""
        timer = ScaledTimer()
        assert isinstance(timer, Timer)


# =============================================================================
# FastForwardTimer Tests
# =============================================================================


class TestFastForwardTimer:
    """Tests for FastForwardTimer."""

    def test_time_ns_from_clock(self) -> None:
        """time_ns returns clock's current time."""
        clock = VirtualClock(start_time=12345)
        timer = FastForwardTimer(clock)
        assert timer.time_ns() == 12345

    @pytest.mark.asyncio
    async def test_sleep_schedules_wakeup(self) -> None:
        """sleep schedules a wakeup on the clock."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)

        async def sleeper() -> None:
            await timer.sleep(1.0)  # 1 second = 1_000_000_000 ns

        task = asyncio.create_task(sleeper())
        await asyncio.sleep(0)

        assert clock.pending_count() == 1
        assert clock.next_wakeup() == 1_000_000_000

        await clock.advance_to(1_000_000_000)
        await task

    @pytest.mark.asyncio
    async def test_sleep_zero_returns_immediately(self) -> None:
        """sleep(0) returns immediately without scheduling."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        await timer.sleep(0)
        assert clock.pending_count() == 0

    @pytest.mark.asyncio
    async def test_sleep_until_delegates_to_clock(self) -> None:
        """sleep_until delegates to clock.sleep_until."""
        clock = VirtualClock(start_time=100)
        timer = FastForwardTimer(clock)

        # Past time - immediate
        await asyncio.wait_for(timer.sleep_until(50), timeout=0.1)

        # Future time - blocks
        async def sleeper() -> None:
            await timer.sleep_until(500)

        task = asyncio.create_task(sleeper())
        await asyncio.sleep(0)
        assert clock.pending_count() == 1

        await clock.advance_to(500)
        await task

    def test_conforms_to_protocol(self) -> None:
        """FastForwardTimer conforms to Timer protocol."""
        clock = VirtualClock()
        timer = FastForwardTimer(clock)
        assert isinstance(timer, Timer)


# =============================================================================
# Periodic Tests
# =============================================================================


class TestPeriodic:
    """Tests for the periodic async iterator."""

    @pytest.mark.asyncio
    async def test_periodic_yields_timestamps(self) -> None:
        """periodic yields timestamps at intervals."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)

        timestamps: list[int] = []

        async def collector() -> None:
            async for t in timer.periodic(1.0):
                timestamps.append(t)
                if len(timestamps) >= 3:
                    break

        task = asyncio.create_task(collector())
        await asyncio.sleep(0)

        # Advance through 3 periods
        for target in [1_000_000_000, 2_000_000_000, 3_000_000_000]:
            await clock.advance_to(target)
            await asyncio.sleep(0)

        await task

        assert timestamps == [1_000_000_000, 2_000_000_000, 3_000_000_000]

    @pytest.mark.asyncio
    async def test_periodic_no_drift(self) -> None:
        """periodic doesn't drift - uses absolute times."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)

        timestamps: list[int] = []

        async def collector() -> None:
            async for t in timer.periodic(1.0):
                timestamps.append(t)
                if len(timestamps) >= 3:
                    break

        task = asyncio.create_task(collector())
        await asyncio.sleep(0)

        # Advance past the tick times (simulating late processing)
        # periodic schedules at absolute times: 1s, 2s, 3s
        # Even if we overshoot, next tick is still at the scheduled time
        await clock.advance_to(1_500_000_000)  # 1.5s - wakes tick@1s
        await asyncio.sleep(0)
        await clock.advance_to(2_500_000_000)  # 2.5s - wakes tick@2s
        await asyncio.sleep(0)
        await clock.advance_to(3_500_000_000)  # 3.5s - wakes tick@3s
        await asyncio.sleep(0)

        await task

        # Ticks fire at their scheduled times (1s, 2s, 3s), not when we advanced
        # But time_ns() returns current clock time when yield happens
        assert timestamps == [1_000_000_000, 2_000_000_000, 3_000_000_000]


# =============================================================================
# Factory Tests
# =============================================================================


class TestCreateTimer:
    """Tests for create_timer factory."""

    def test_default_is_scaled_timer(self) -> None:
        """Default creates ScaledTimer with speed=1.0."""
        timer = create_timer()
        assert isinstance(timer, ScaledTimer)

    def test_speed_creates_scaled_timer(self) -> None:
        """Non-inf speed creates ScaledTimer."""
        timer = create_timer(speed=2.0)
        assert isinstance(timer, ScaledTimer)

    def test_inf_speed_requires_clock(self) -> None:
        """speed=inf without clock raises ValueError."""
        with pytest.raises(ValueError, match="requires a VirtualClock"):
            create_timer(speed=float("inf"))

    def test_inf_speed_with_clock(self) -> None:
        """speed=inf with clock creates FastForwardTimer."""
        clock = VirtualClock()
        timer = create_timer(speed=float("inf"), clock=clock)
        assert isinstance(timer, FastForwardTimer)

    def test_start_time_passed_to_scaled_timer(self) -> None:
        """start_time is passed to ScaledTimer."""
        timer = create_timer(speed=1.0, start_time=1000)
        assert isinstance(timer, ScaledTimer)
        # Time should be close to start_time
        assert abs(timer.time_ns() - 1000) < 10_000_000


# =============================================================================
# Integration Tests
# =============================================================================


class TestTimerIntegration:
    """Integration tests for timer system."""

    @pytest.mark.asyncio
    async def test_deterministic_ordering(self) -> None:
        """Fast-forward mode ensures deterministic timer/message ordering."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        events: list[str] = []

        async def periodic_task() -> None:
            """Task that logs every 100ms."""
            async for t in timer.periodic(0.1):
                events.append(f"tick@{t // 1_000_000}")
                if t >= 500_000_000:
                    break

        async def message_processor() -> None:
            """Simulate processing messages at specific times."""
            messages = [
                (50_000_000, "msg1"),
                (150_000_000, "msg2"),
                (250_000_000, "msg3"),
            ]
            for msg_time, msg in messages:
                await clock.advance_to(msg_time)
                events.append(f"{msg}@{msg_time // 1_000_000}")

        # Start periodic task
        periodic = asyncio.create_task(periodic_task())
        await asyncio.sleep(0)

        # Process messages (this drives time forward)
        await message_processor()

        # Advance to let periodic complete
        await clock.advance_to(600_000_000)
        await periodic

        # Events should be in timestamp order:
        # msg1@50, tick@100, msg2@150, tick@200, msg3@250, tick@300, tick@400, tick@500
        expected = [
            "msg1@50",
            "tick@100",
            "msg2@150",
            "tick@200",
            "msg3@250",
            "tick@300",
            "tick@400",
            "tick@500",
        ]
        assert events == expected

    @pytest.mark.asyncio
    async def test_multiple_periodic_timers(self) -> None:
        """Multiple periodic timers interleave correctly."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        events: list[str] = []

        async def fast_ticker() -> None:
            count = 0
            async for t in timer.periodic(0.1):  # 100ms
                events.append(f"fast@{t // 1_000_000}")
                count += 1
                if count >= 4:
                    break

        async def slow_ticker() -> None:
            count = 0
            async for t in timer.periodic(0.25):  # 250ms
                events.append(f"slow@{t // 1_000_000}")
                count += 1
                if count >= 2:
                    break

        fast = asyncio.create_task(fast_ticker())
        slow = asyncio.create_task(slow_ticker())
        await asyncio.sleep(0)

        # Advance time to complete both
        await clock.advance_to(500_000_000)

        await asyncio.gather(fast, slow)

        # Check ordering
        expected = [
            "fast@100",
            "fast@200",
            "slow@250",
            "fast@300",
            "fast@400",
            "slow@500",
        ]
        assert events == expected
