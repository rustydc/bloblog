"""Tests for runtime module (core graph operations)."""

import pytest
from .test_utils import StringCodec, make_consumer_node, make_producer_node

from tinman import In, Out, get_node_specs, validate_nodes
from typing import Annotated


class TestGetNodeSpecs:
    def test_extracts_inputs_and_outputs(self):
        """Test that get_node_specs correctly parses node signatures."""
        producer = make_producer_node("producer", ["msg"])
        consumer, _ = make_consumer_node("producer_output", 1)
        
        specs = get_node_specs([producer, consumer])
        
        assert len(specs) == 2
        
        # Producer should have one output, no inputs
        producer_spec = specs[0]
        assert len(producer_spec.inputs) == 0
        assert len(producer_spec.outputs) == 1
        assert "producer_output" in [ch for _, (ch, _) in producer_spec.outputs.items()]
        
        # Consumer should have one input, no outputs
        consumer_spec = specs[1]
        assert len(consumer_spec.inputs) == 1
        assert len(consumer_spec.outputs) == 0
        assert "producer_output" in [ch for _, (ch, _) in consumer_spec.inputs.items()]
    
    def test_extracts_codecs(self):
        """Test that codecs are correctly extracted."""
        producer = make_producer_node("test", ["hi"])
        specs = get_node_specs([producer])
        
        # Should have StringCodec in outputs
        for _, (channel_name, codec) in specs[0].outputs.items():
            assert isinstance(codec, StringCodec)
    
    def test_detects_dict_injection(self):
        """Test that get_node_specs correctly detects dict[str, In] parameters."""
        async def logging_node(channels: dict[str, In]) -> None:
            pass
        
        specs = get_node_specs([logging_node])
        assert len(specs) == 1
        assert specs[0].all_channels_param == "channels"
        assert len(specs[0].inputs) == 0
        assert len(specs[0].outputs) == 0


class TestValidateNodes:
    def test_valid_simple_pipeline(self):
        producer = make_producer_node("producer", ["msg"])
        consumer, _ = make_consumer_node("producer_output", 1)

        # Should not raise
        validate_nodes([producer, consumer])

    def test_duplicate_output_channel_raises(self):
        producer1 = make_producer_node("producer", ["msg"])
        producer2 = make_producer_node("producer", ["msg"])  # Same output channel name

        with pytest.raises(ValueError, match="produced by multiple nodes"):
            validate_nodes([producer1, producer2])

    def test_missing_input_producer_raises(self):
        consumer, _ = make_consumer_node("nonexistent_channel", 1)

        with pytest.raises(ValueError, match="has no producer node"):
            validate_nodes([consumer])

    def test_producer_only_is_valid(self):
        producer = make_producer_node("producer", ["msg"])

        # Should not raise - producer with no consumers is valid
        validate_nodes([producer])
    
    def test_skips_dict_injection_nodes(self):
        """Test that validation doesn't require producers for dict[str, In] nodes."""
        async def producer(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            pass
        
        async def logger(channels: dict[str, In]) -> None:
            pass
        
        # Should not raise - logger node doesn't need explicit inputs
        validate_nodes([producer, logger])

    def test_detects_timer_param(self):
        """Test that get_node_specs detects Timer parameters."""
        from tinman import Timer
        
        async def timed_node(timer: Timer) -> None:
            pass
        
        specs = get_node_specs([timed_node])
        assert len(specs) == 1
        assert specs[0].timer_param == "timer"

    def test_rejects_multiple_timer_params(self):
        """Test that multiple Timer parameters raise an error."""
        from tinman import Timer
        
        async def bad_node(timer1: Timer, timer2: Timer) -> None:
            pass
        
        with pytest.raises(ValueError, match="multiple Timer"):
            get_node_specs([bad_node])


class TestTimerInjection:
    """Tests for Timer parameter injection in run_nodes."""
    
    @pytest.mark.asyncio
    async def test_timer_injected_into_node(self):
        """Test that Timer is injected into nodes that request it."""
        from tinman import Timer
        from tinman.runtime import run_nodes
        
        received_timer = None
        
        async def timed_producer(
            timer: Timer,
            out: Annotated[Out[str], "data", StringCodec()]
        ) -> None:
            nonlocal received_timer
            received_timer = timer
            await out.publish("hello")
        
        consumer, results = make_consumer_node("data", 1)
        await run_nodes([timed_producer, consumer])
        
        assert received_timer is not None
        assert isinstance(received_timer, Timer)
    
    @pytest.mark.asyncio
    async def test_custom_timer_injected(self):
        """Test that a custom Timer can be provided to run_nodes."""
        from tinman import Timer, VirtualClock, FastForwardTimer
        from tinman.runtime import run_nodes
        
        clock = VirtualClock(start_time=1000)
        custom_timer = FastForwardTimer(clock)
        received_timer = None
        
        async def timed_producer(
            timer: Timer,
            out: Annotated[Out[str], "data", StringCodec()]
        ) -> None:
            nonlocal received_timer
            received_timer = timer
            await out.publish("hello")
        
        consumer, results = make_consumer_node("data", 1)
        await run_nodes([timed_producer, consumer], timer=custom_timer)
        
        assert received_timer is custom_timer
    
    @pytest.mark.asyncio
    async def test_timer_time_ns_works(self):
        """Test that the injected timer's time_ns works."""
        from tinman import Timer
        from tinman.runtime import run_nodes
        
        recorded_time = None
        
        async def timed_producer(
            timer: Timer,
            out: Annotated[Out[str], "data", StringCodec()]
        ) -> None:
            nonlocal recorded_time
            recorded_time = timer.time_ns()
            await out.publish("hello")
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([timed_producer, consumer])
        
        assert recorded_time is not None
        assert recorded_time > 0
    
    @pytest.mark.asyncio
    async def test_timer_sleep_works(self):
        """Test that the injected timer's sleep works."""
        from tinman import Timer, VirtualClock, FastForwardTimer
        from tinman.runtime import run_nodes
        import asyncio
        
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        sleep_completed = False
        
        async def timed_producer(
            timer: Timer,
            out: Annotated[Out[str], "data", StringCodec()]
        ) -> None:
            nonlocal sleep_completed
            # Schedule a sleep
            sleep_task = asyncio.create_task(timer.sleep(1.0))
            await asyncio.sleep(0)  # Let it start
            # Advance the clock
            await clock.advance_to(1_000_000_000)
            await sleep_task
            sleep_completed = True
            await out.publish("hello")
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([timed_producer, consumer], timer=timer)
        
        assert sleep_completed
