"""Simple example demonstrating node running.

Run with: uv run python -m examples.simple
"""
import asyncio
from pathlib import Path
from typing import Annotated
from tempfile import TemporaryDirectory

from bloblog import In, Out, run, playback, enable_pickle_codec


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
        print(f"Counter: Starting...")
        async for msg in messages:
            self.count += 1
            counted = f"[{self.prefix} {self.count}] {msg}"
            print(f"Counter: {msg} -> {counted}")
            await counted_out.publish(counted)
        print(f"Counter: Done ({self.count} messages)")


async def main():
    """Run example pipelines."""
    enable_pickle_codec()

    with TemporaryDirectory(delete=False) as log_dir:
        log_path = Path(log_dir)
    
        print("Example 1: Simple Pipeline (Producer -> Uppercase -> Consumer)")

        await run([producer, uppercase, consumer])
        
        print("\n\nExample 2: With Stateful Node (Producer -> Counter)")

        counter = Counter(prefix="Item")
        
        # Consumer for counted messages
        async def consumer2(input: Annotated[In[str], "counted"]):
            async for msg in input:
                print(f"Consumer2: {msg}")
        
        await run([producer, counter.run, consumer2], log_dir=log_path)

        print("\n\nExample 3: Playback")

        await playback([consumer2], playback_dir=log_path)

        print("\n\nExample 4: Playback (4.0x)")
        await playback([consumer2], playback_dir=log_path, speed=4.0)

        print("Done!")
        print(f"Logs stored in: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
