"""Tests for playback functionality."""
import pytest

from bloblog import Node, Input, Output, run_nodes, make_playback_nodes, playback_nodes
from test_utils import StringCodec


class TestPlayback:
    @pytest.mark.asyncio
    async def test_playback_from_logs(self, tmp_path):
        """Test that non-live nodes play back from logs."""
        messages = ["hello", "world", "test"]
        
        # First run: record producer output
        class RecordingProducer(Node):
            output: Output[str] = Output("producer_output", StringCodec())
            
            async def run(self) -> None:
                for msg in messages:
                    await self.output.publish(msg)
        
        await run_nodes([RecordingProducer()], log_dir=tmp_path)
        
        # Verify log file was created
        log_file = tmp_path / "producer_output.bloblog"
        assert log_file.exists()
        
        # Second run: playback producer (implicitly), run consumer live
        received: list[str] = []
        
        class LiveConsumer(Node):
            input: Input[str] = Input("producer_output", StringCodec())
            
            async def run(self) -> None:
                async for msg in self.input:
                    received.append(msg)
                    if len(received) >= len(messages):
                        break
        
        # playback_nodes adds log players for producer_output and runs everything
        consumer = LiveConsumer()
        output_dir = tmp_path / "output"
        await playback_nodes([consumer], playback_dir=tmp_path, log_dir=output_dir)
        
        assert received == messages

    @pytest.mark.asyncio
    async def test_playback_nodes_validates_log_exists(self, tmp_path):
        """Test that make_playback_nodes raises if log file doesn't exist."""
        class ConsumerOfMissing(Node):
            input: Input[str] = Input("missing_channel", StringCodec())
            async def run(self) -> None:
                pass
        
        with pytest.raises(FileNotFoundError, match="No log file for channel"):
            make_playback_nodes([ConsumerOfMissing()], playback_dir=tmp_path)
