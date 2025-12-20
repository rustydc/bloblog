"""Simple example demonstrating node running with logging and playback.

Usage:
    # Run pipeline and log output
    tinman run examples.simple:pipeline --log-dir logs/ --pickle

    # Playback through consumer
    tinman playback examples.simple:consumer --from logs/ --pickle

    # Use factory to create configured node
    tinman playback examples.simple:create_counter --from logs/ --pickle
"""

import asyncio
from typing import Annotated

from tinman import In, Out


# Example 1: Simple function node
async def producer(output: Annotated[Out[str], "messages"]):
    for i in range(1, 6):
        msg = f"Message {i}"
        print(f"Producer: Sending '{msg}'")
        await output.publish(msg)
        await asyncio.sleep(0.5)


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
    async for msg in input:
        print(f"Consumer: Received '{msg}'")

# Pre-built pipeline for CLI usage
pipeline = [producer, uppercase, consumer]


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


# Factory function for CLI - creates configured Counter node
def create_counter():
    """Factory that creates a Counter node with custom prefix."""
    return Counter(prefix="Characters:").run
