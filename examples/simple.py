"""Simple example demonstrating node running.

Run with: python3 -m examples.simple
"""
import asyncio
import sys
from pathlib import Path
from collections.abc import Buffer
from typing import Annotated
from tempfile import TemporaryDirectory

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bloblog import In, Out, run_nodes, Codec, playback_nodes


class StringCodec(Codec[str]):
    """Simple codec for string data."""
    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")
    
    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


# Example 1: Simple function node
async def producer(
    output: Annotated[Out[str], "messages", StringCodec()],
):
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
    output: Annotated[Out[str], "uppercase", StringCodec()],
):
    """Transform messages to uppercase."""
    print("Uppercase: Starting...")
    async for msg in input:
        upper = msg.upper()
        print(f"Uppercase: {msg} -> {upper}")
        await output.publish(upper)
    print("Uppercase: Done")


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
        input: Annotated[In[str], "messages"],
        output: Annotated[Out[str], "counted", StringCodec()],
    ):
        """Add counter to each message."""
        print(f"Counter: Starting...")
        async for msg in input:
            self.count += 1
            counted = f"[{self.prefix} {self.count}] {msg}"
            print(f"Counter: {msg} -> {counted}")
            await output.publish(counted)
        print(f"Counter: Done ({self.count} messages)")


async def main():
    """Run example pipelines."""

    with TemporaryDirectory() as log_dir:
        log_path = Path(log_dir)
    
        print("Example 1: Simple Pipeline (Producer -> Uppercase -> Consumer)")

        await run_nodes([producer, uppercase, consumer])
        
        print("\n\nExample 2: With Stateful Node (Producer -> Counter)")

        counter = Counter(prefix="Item")
        
        # Consumer for counted messages
        async def consumer2(input: Annotated[In[str], "counted"]):
            async for msg in input:
                print(f"Consumer2: {msg}")
        
        await run_nodes([producer, counter.run, consumer2], log_dir=log_path)

        print("\n\nExample 3: Playback")

        await playback_nodes([consumer2], playback_dir=log_path)

        print("\n\nExample 4: Playback (4.0x)")
        await playback_nodes([consumer2], playback_dir=log_path, playback_speed=4.0)

        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
