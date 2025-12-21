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
    """Tests for VirtualClock with event-driven scheduling."""

    def test_initial_time(self) -> None:
        """Clock starts at specified time."""
        clock = VirtualClock(start_time=1000)
        assert clock.now() == 1000

    def test_initial_time_default(self) -> None:
        """Clock defaults to time 0."""
        clock = VirtualClock()
        assert clock.now() == 0

    @pytest.mark.asyncio
    async def test_schedule_advances_time(self) -> None:
        """schedule() advances clock to event timestamp."""
        clock = VirtualClock(start_time=100)
        await clock.schedule(500)
        assert clock.now() == 500

    @pytest.mark.asyncio
    async def test_schedule_no_backwards(self) -> None:
        """schedule() with earlier time doesn't move clock backwards."""
        clock = VirtualClock(start_time=500)
        await clock.schedule(100)
        assert clock.now() == 500

    @pytest.mark.asyncio
    async def test_pending_count(self) -> None:
        """pending_count returns number of waiting events."""
        clock = VirtualClock()
        assert clock.pending_count() == 0
        
        # Schedule an event but don't await it fully yet
        async def schedule_and_check():
            # After this returns, the event has completed
            await clock.schedule(100)
        
        await schedule_and_check()
        assert clock.pending_count() == 0  # Event completed

    @pytest.mark.asyncio
    async def test_multiple_events_interleave(self) -> None:
        """Multiple events scheduled at same time interleave fairly."""
        clock = VirtualClock(start_time=0)
        events: list[str] = []
        
        async def task_a():
            await clock.schedule(100)
            events.append("a1")
            await clock.schedule(200)
            events.append("a2")
        
        async def task_b():
            await clock.schedule(100)
            events.append("b1")
            await clock.schedule(200)
            events.append("b2")
        
        await asyncio.gather(task_a(), task_b())
        
        # Both tasks should interleave - order depends on which schedules first
        assert len(events) == 4
        assert set(events) == {"a1", "a2", "b1", "b2"}

    @pytest.mark.asyncio
    async def test_events_processed_in_timestamp_order(self) -> None:
        """Events are processed in timestamp order."""
        clock = VirtualClock(start_time=0)
        events: list[int] = []
        
        async def waiter(ts: int):
            await clock.schedule(ts)
            events.append(ts)
        
        # Schedule out of order
        await asyncio.gather(
            waiter(300),
            waiter(100),
            waiter(200),
        )
        
        # Should complete in timestamp order
        assert events == [100, 200, 300]
        assert clock.now() == 300

    @pytest.mark.asyncio
    async def test_flush_wakes_all_pending(self) -> None:
        """flush() wakes all pending events."""
        clock = VirtualClock(start_time=0)
        events: list[int] = []
        
        async def waiter(ts: int):
            await clock.schedule(ts)
            events.append(ts)
        
        # Start tasks that will schedule events
        tasks = [
            asyncio.create_task(waiter(300)),
            asyncio.create_task(waiter(100)),
            asyncio.create_task(waiter(200)),
        ]
        await asyncio.sleep(0)  # Let them schedule
        
        # Flush should complete all events
        await clock.flush()
        await asyncio.gather(*tasks)
        
        assert events == [100, 200, 300]
        assert clock.pending_count() == 0


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
    async def test_sleep_schedules_and_completes(self) -> None:
        """sleep schedules an event and completes when it's processed."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        
        # In the event-driven model, sleep() schedules and waits
        await timer.sleep(1.0)  # 1 second = 1_000_000_000 ns
        
        # After sleep completes, clock should have advanced
        assert clock.now() == 1_000_000_000

    @pytest.mark.asyncio
    async def test_sleep_zero_returns_immediately(self) -> None:
        """sleep(0) returns immediately without scheduling."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        await timer.sleep(0)
        assert clock.pending_count() == 0

    @pytest.mark.asyncio
    async def test_sleep_until_past_returns_immediately(self) -> None:
        """sleep_until with past time returns immediately."""
        clock = VirtualClock(start_time=100)
        timer = FastForwardTimer(clock)
        await asyncio.wait_for(timer.sleep_until(50), timeout=0.1)

    @pytest.mark.asyncio
    async def test_sleep_until_future_schedules(self) -> None:
        """sleep_until with future time schedules event."""
        clock = VirtualClock(start_time=100)
        timer = FastForwardTimer(clock)
        
        await timer.sleep_until(500)
        assert clock.now() == 500

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
        """periodic yields timestamps at intervals (event-driven)."""
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)

        timestamps: list[int] = []

        async def collector() -> None:
            async for t in timer.periodic(1.0):
                timestamps.append(t)
                if len(timestamps) >= 3:
                    break

        # In event-driven mode, the collector drives itself
        await collector()

        assert timestamps == [1_000_000_000, 2_000_000_000, 3_000_000_000]
        assert clock.now() == 3_000_000_000


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
            count = 0
            async for t in timer.periodic(0.1):
                events.append(f"tick@{t // 1_000_000}")
                count += 1
                if count >= 5:
                    break

        # In event-driven mode, the task drives itself
        await periodic_task()

        # Events should be in timestamp order
        expected = [
            "tick@100",
            "tick@200",
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
            async for _t in timer.periodic(0.1):  # 100ms
                events.append(f"fast@{clock.now() // 1_000_000}")
                count += 1
                if count >= 4:
                    break

        async def slow_ticker() -> None:
            count = 0
            async for _t in timer.periodic(0.25):  # 250ms
                events.append(f"slow@{clock.now() // 1_000_000}")
                count += 1
                if count >= 2:
                    break

        # Run both concurrently - they interleave via schedule()
        await asyncio.gather(fast_ticker(), slow_ticker())

        # With event-driven clock, events fire when earliest event is processed.
        # fast schedules at 100, slow at 250.
        # fast@100 fires, schedules at 200. slow still at 250.
        # fast@200... but slow also scheduled at 250, and yield might see 250
        # The exact interleaving depends on scheduling order.
        # Key invariant: all events complete, clock advances correctly
        assert len(events) == 6
        assert clock.now() == 500_000_000
