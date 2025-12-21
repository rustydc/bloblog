"""Tests for runtime module (core graph operations)."""

import asyncio
import pytest
from .test_utils import StringCodec, make_consumer_node, make_producer_node

from tinman import In, Out, get_node_specs, validate_nodes, daemon
from tinman.oblog import PickleCodec
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

    def test_assigns_unique_names(self):
        """Test that get_node_specs assigns unique names to nodes."""
        async def producer(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            pass
        
        async def consumer(inp: Annotated[In[str], "data"]) -> None:
            pass
        
        specs = get_node_specs([producer, consumer])
        
        assert specs[0].name == "producer"
        assert specs[1].name == "consumer"

    def test_disambiguates_duplicate_names(self):
        """Test that duplicate function names get _1, _2 suffixes."""
        async def worker(out: Annotated[Out[str], "data1", StringCodec()]) -> None:
            pass
        
        # Create lambdas with same __name__ by reassigning
        worker1 = worker
        
        async def worker(out: Annotated[Out[str], "data2", StringCodec()]) -> None:
            pass
        worker2 = worker
        
        async def worker(out: Annotated[Out[str], "data3", StringCodec()]) -> None:
            pass
        worker3 = worker
        
        specs = get_node_specs([worker1, worker2, worker3])
        
        assert specs[0].name == "worker_1"
        assert specs[1].name == "worker_2"
        assert specs[2].name == "worker_3"

    def test_preserves_explicit_names(self):
        """Test that explicit names in NodeSpec are preserved."""
        from tinman import NodeSpec
        
        async def my_func(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            pass
        
        spec = NodeSpec(
            node_fn=my_func,
            inputs={},
            outputs={"out": ("data", StringCodec())},
            name="custom_name"
        )
        
        specs = get_node_specs([spec])
        assert specs[0].name == "custom_name"

    def test_disambiguates_mixed_explicit_and_derived(self):
        """Test disambiguation when explicit names conflict with derived names."""
        from tinman import NodeSpec
        
        async def worker(out: Annotated[Out[str], "data1", StringCodec()]) -> None:
            pass
        
        explicit_spec = NodeSpec(
            node_fn=worker,
            inputs={},
            outputs={"out": ("data2", StringCodec())},
            name="worker"  # Same as derived name
        )
        
        specs = get_node_specs([worker, explicit_spec])
        
        # Both have base name "worker", so both get suffixes
        assert specs[0].name == "worker_1"
        assert specs[1].name == "worker_2"


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
        """Test that the injected timer's sleep works with event-driven clock."""
        from tinman import Timer, VirtualClock, FastForwardTimer
        from tinman.runtime import run_nodes
        import asyncio
        
        clock = VirtualClock(start_time=0)
        timer = FastForwardTimer(clock)
        sleep_completed = False
        time_after_sleep = None
        
        async def timed_producer(
            timer: Timer,
            out: Annotated[Out[str], "data", StringCodec()]
        ) -> None:
            nonlocal sleep_completed, time_after_sleep
            # Sleep should schedule an event and wait
            await timer.sleep(1.0)
            sleep_completed = True
            time_after_sleep = timer.time_ns()
            await out.publish("hello")
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([timed_producer, consumer], timer=timer, clock=clock)
        
        assert sleep_completed
        # After sleep(1.0), clock should have advanced to 1 second
        assert time_after_sleep == 1_000_000_000


class TestNodeNames:
    """Tests for node naming and get_current_node_name."""
    
    @pytest.mark.asyncio
    async def test_get_current_node_name_returns_name_during_execution(self):
        """Test that get_current_node_name returns the node's name during execution."""
        from tinman import get_current_node_name
        from tinman.runtime import run_nodes
        
        captured_name = None
        
        async def my_producer(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            nonlocal captured_name
            captured_name = get_current_node_name()
            await out.publish("hello")
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([my_producer, consumer], install_signal_handlers=False)
        
        assert captured_name == "my_producer"
    
    @pytest.mark.asyncio
    async def test_get_current_node_name_returns_none_outside_node(self):
        """Test that get_current_node_name returns None outside node execution."""
        from tinman import get_current_node_name
        
        assert get_current_node_name() is None
    
    @pytest.mark.asyncio
    async def test_get_current_node_name_with_disambiguated_names(self):
        """Test that disambiguated names are reflected in get_current_node_name."""
        from tinman import get_current_node_name
        from tinman.runtime import run_nodes
        
        captured_names: list[str | None] = []
        
        async def worker(out: Annotated[Out[str], "data1", StringCodec()]) -> None:
            captured_names.append(get_current_node_name())
            await out.publish("hello")
        worker1 = worker
        
        async def worker(out: Annotated[Out[str], "data2", StringCodec()]) -> None:
            captured_names.append(get_current_node_name())
            await out.publish("hello")
        worker2 = worker
        
        async def final_consumer(
            inp1: Annotated[In[str], "data1"],
            inp2: Annotated[In[str], "data2"]
        ) -> None:
            async for _ in inp1:
                pass
            async for _ in inp2:
                pass
        
        await run_nodes([worker1, worker2, final_consumer], install_signal_handlers=False)
        
        # Should have captured worker_1 and worker_2
        assert "worker_1" in captured_names
        assert "worker_2" in captured_names


class TestDaemonNodes:
    @pytest.mark.asyncio
    async def test_daemon_node_cancelled_when_main_completes(self):
        """Test that daemon nodes are cancelled when main nodes complete."""
        from tinman.runtime import run_nodes, NodeSpec
        from tinman.oblog import PickleCodec
        
        daemon_cancelled = False
        main_completed = False
        
        async def main_node(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            nonlocal main_completed
            await out.publish("hello")
            main_completed = True
        
        async def daemon_node(out: Annotated[Out[str], "daemon_data", StringCodec()]) -> None:
            nonlocal daemon_cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                daemon_cancelled = True
                raise
        
        daemon_spec = NodeSpec(
            node_fn=daemon_node,
            inputs={},
            outputs={"out": ("daemon_data", StringCodec())},
            daemon=True,
        )
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([main_node, consumer, daemon_spec])
        
        assert main_completed
        assert daemon_cancelled

    @pytest.mark.asyncio
    async def test_daemon_decorator(self):
        """Test that @daemon decorator marks nodes as daemons."""
        from tinman.runtime import run_nodes, daemon
        
        daemon_cancelled = False
        main_completed = False
        
        async def main_node(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            nonlocal main_completed
            await out.publish("hello")
            main_completed = True
        
        @daemon
        async def my_daemon(out: Annotated[Out[str], "daemon_data", StringCodec()]) -> None:
            nonlocal daemon_cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                daemon_cancelled = True
                raise
        
        consumer, _ = make_consumer_node("data", 1)
        await run_nodes([main_node, consumer, my_daemon])
        
        assert main_completed
        assert daemon_cancelled


class TestShutdown:
    """Tests for graceful shutdown behavior."""
    
    @pytest.mark.asyncio
    async def test_external_cancellation_cancels_all_nodes(self):
        """Test that external cancellation triggers cleanup of all nodes."""
        from tinman.runtime import run_nodes
        
        main_cancelled = False
        daemon_cancelled = False
        
        async def main_node(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            nonlocal main_cancelled
            try:
                while True:
                    await out.publish("hello")
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                main_cancelled = True
                raise
        
        @daemon
        async def daemon_node(out: Annotated[Out[str], "daemon_data", StringCodec()]) -> None:
            nonlocal daemon_cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                daemon_cancelled = True
                raise
        
        consumer, _ = make_consumer_node("data", 3)
        
        # Start run_nodes and cancel it externally
        task = asyncio.create_task(
            run_nodes([main_node, consumer, daemon_node], install_signal_handlers=False)
        )
        
        # Give it time to start
        await asyncio.sleep(0.05)
        
        # External cancellation
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task
        
        assert main_cancelled
        assert daemon_cancelled
    
    @pytest.mark.asyncio
    async def test_node_error_cancels_other_nodes(self):
        """Test that an error in one node cancels others."""
        from tinman.runtime import run_nodes
        
        other_cancelled = False
        
        async def error_node(out: Annotated[Out[str], "error_data", StringCodec()]) -> None:
            await asyncio.sleep(0.01)
            raise ValueError("intentional error")
        
        async def slow_node(out: Annotated[Out[str], "slow_data", StringCodec()]) -> None:
            nonlocal other_cancelled
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                other_cancelled = True
                raise
        
        with pytest.raises(ValueError, match="intentional error"):
            await run_nodes([error_node, slow_node], install_signal_handlers=False)
        
        # The other node should have been cancelled during cleanup
        assert other_cancelled
    
    @pytest.mark.asyncio
    async def test_shutdown_timeout_parameter(self):
        """Test that shutdown_timeout parameter is accepted."""
        from tinman.runtime import run_nodes
        
        async def quick_node(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            await out.publish("done")
        
        consumer, _ = make_consumer_node("data", 1)
        
        # Should complete without error with custom timeout
        await run_nodes(
            [quick_node, consumer], 
            install_signal_handlers=False,
            shutdown_timeout=1.0
        )
    
    @pytest.mark.asyncio
    async def test_install_signal_handlers_false_in_tests(self):
        """Test that signal handlers can be disabled (important for tests)."""
        from tinman.runtime import run_nodes
        
        async def quick_node(out: Annotated[Out[str], "data", StringCodec()]) -> None:
            await out.publish("done")
        
        consumer, _ = make_consumer_node("data", 1)
        
        # Should work without signal handlers
        await run_nodes([quick_node, consumer], install_signal_handlers=False)