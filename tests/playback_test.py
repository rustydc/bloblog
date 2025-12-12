"""Tests for playback functionality."""

from typing import Annotated

import pytest
from test_utils import StringCodec

from tinman import In, Out, run


class TestPlayback:
    @pytest.mark.asyncio
    async def test_playback_from_logs(self, tmp_path):
        """Test that non-live nodes play back from logs."""
        messages = ["hello", "world", "test"]

        # First run: record producer output
        codec = StringCodec()

        async def recording_producer(output: Annotated[Out[str], "producer_output", codec]) -> None:
            for msg in messages:
                await output.publish(msg)

        await run([recording_producer], log_dir=tmp_path)

        # Verify log file was created
        log_file = tmp_path / "producer_output.blog"
        assert log_file.exists()

        # Second run: playback producer (implicitly), run consumer live
        received: list[str] = []

        async def live_consumer(input: Annotated[In[str], "producer_output"]) -> None:
            async for msg in input:
                received.append(msg)
                if len(received) >= len(messages):
                    break

        # run with playback adds log players for producer_output and runs everything
        output_dir = tmp_path / "output"
        await run([live_consumer], playback_dir=tmp_path, log_dir=output_dir)

        assert received == messages

    @pytest.mark.asyncio
    async def test_playback_validates_log_exists(self, tmp_path):
        """Test that run with playback raises if log file doesn't exist."""

        async def consumer_of_missing(input: Annotated[In[str], "missing_channel"]) -> None:
            pass

        with pytest.raises(FileNotFoundError, match="No log file for channel"):
            await run([consumer_of_missing], playback_dir=tmp_path)
