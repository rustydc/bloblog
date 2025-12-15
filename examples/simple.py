"""Simple example demonstrating node running with logging and playback.

Run with: uv run python -m examples.simple
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from tinman import In, Out, enable_pickle_codec, run, playback


# Example 1: Simple function node
async def producer(output: Annotated[Out[str], "messages"]):
    """Producer node - just a function!"""
    print("Producer: Starting...")
    for i in range(5):
        msg = f"Message {i}"
        print(f"Producer: Sending '{msg}'")
        await output.publish(msg)
        await asyncio.sleep(0.5)
    print("Producer: Done")


# Example 2: Transform node
async def uppercase(
    input: Annotated[In[str], "messages"],
    output: Annotated[Out[str], "uppercase"],
):
    async for msg in input:
        await output.publish(msg.upper())


# Example 3: Consumer node
async def consumer(
    input: Annotated[In[str], "uppercase"],
):
    """Print received messages."""
    print("Consumer: Starting...")
    async for msg in input:
        print(f"Consumer: Received '{msg}'")
    print("Consumer: Done")


# Example 4: Stateful class-based node
class Counter:
    """Node with state."""

    def __init__(self, prefix: str = "Count"):
        self.prefix = prefix
        self.count = 0

    async def run(
        self,
        messages: Annotated[In[str], "messages"],
        counted_out: Annotated[Out[str], "counted"],
    ):
        """Add counter to each message."""
        print("Counter: Starting...")
        async for msg in messages:
            self.count += len(msg)
            counted = f"[{self.prefix} {self.count}] {msg}"
            print(f"Counter: {msg} characters -> {counted}")
            await counted_out.publish(counted)
        print(f"Counter: Done ({self.count} messages)")


async def main():
    """Run example pipelines."""
    enable_pickle_codec()

    with TemporaryDirectory(delete=False) as log_dir:
        log_path = Path(log_dir)

        print("Example 1: Simple Pipeline (Producer -> Uppercase -> Consumer)")
        print("-" * 70)

        ex1_log = log_path / "ex1"
        await run([producer, uppercase, consumer], log_dir=ex1_log)

        print("\n\nExample 2: Playback with different processor")
        print("-" * 70)

        counter = Counter(prefix="Characters:")

        # Consumer for counted messages
        async def consumer2(input: Annotated[In[str], "counted"]):
            async for msg in input:
                print(f"Consumer2: {msg}")

        # Playback "messages", run through counter (stateful), log "counted" output
        ex2_log = log_path / "ex2"
        await playback([counter.run, consumer2], playback_dir=ex1_log, log_dir=ex2_log, speed=1.0)

        print("\n\nExample 3: Playback at 5.0x Speed")
        print("-" * 70)

        await playback([consumer2], playback_dir=ex2_log, speed=5.0)

        print("\n\nExample 4: Playback at full speed")
        print("-" * 70)

        await playback([consumer2], playback_dir=ex2_log)

        print("\n" + "=" * 70)
        print("Done!")
        print(f"Logs stored in: {log_path}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
