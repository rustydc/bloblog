"""Simple example demonstrating node running with explicit logging and playback.

Run with: uv run python -m examples.simple
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from tinman import In, Out, enable_pickle_codec, run
from tinman.run import create_logging_node, create_playback_graph, get_node_specs


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
        print("-" * 70)

        await run([producer, uppercase, consumer])

        print("\n\nExample 2: With Stateful Node and Explicit Logging")
        print("-" * 70)

        counter = Counter(prefix="Item")

        # Consumer for counted messages
        async def consumer2(input: Annotated[In[str], "counted"]):
            async for msg in input:
                print(f"Consumer2: {msg}")

        # Explicit logging workflow:
        # 1. Get specs to extract channel codecs
        # 2. Create a logger node that subscribes to all channels
        # 3. Run with logger as an additional node
        nodes = [producer, counter.run, consumer2]
        specs = get_node_specs(nodes)
        codecs = {ch: codec for spec in specs for _, (ch, codec) in spec.outputs.items()}
        logger = create_logging_node(log_path, codecs)
        
        await run([*nodes, logger])  # Logger is just another node!

        print("\n\nExample 3: Playback with Explicit Playback Graph")
        print("-" * 70)

        # Explicit playback workflow:
        # 1. create_playback_graph() analyzes what channels are missing
        # 2. Reads codecs from log files
        # 3. Creates playback nodes and returns [playback_node, *original_nodes]
        graph = await create_playback_graph([consumer2], log_path)
        await run(graph)  # Runs consumer2 with playback providing "counted" channel

        print("\n\nExample 4: Playback at 4.0x Speed")
        print("-" * 70)
        
        graph = await create_playback_graph([consumer2], log_path, speed=4.0)
        await run(graph)

        print("\n" + "=" * 70)
        print("Done!")
        print(f"Logs stored in: {log_path}")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
