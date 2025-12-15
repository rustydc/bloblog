# Tinman

**A minimal async framework for building data pipelines with logging and playback.**

Tinman is built from five composable modules:

1. **BlobLog** - Fast binary logging (timestamp + raw bytes)
2. **ObLog** - Object logging with codecs (builds on BlobLog)
3. **PubSub** - Simple async pub/sub channels  
4. **Runtime** - Core execution engine with graph validation
5. **Launcher** - High-level utilities for logging and playback

## Installation

```bash
# Python 3.12+
git clone https://github.com/rustydc/tinman.git
cd tinman
pip install -e .
```

## Quick Start

```python
from typing import Annotated
from tinman import In, Out, run

async def producer(out: Annotated[Out[str], "messages"]):
    for i in range(10):
        await out.publish(f"message {i}")

async def consumer(inp: Annotated[In[str], "messages"]):
    async for msg in inp:
        print(f"Got: {msg}")

await run([producer, consumer])
```

---

## Components

### 1. BlobLog - Binary Logging

Fast, simple binary log format: `(timestamp_ns: u64, length: u64, data: bytes)`.

```python
from pathlib import Path
from tinman.bloblog import BlobLog

# Write
blob = BlobLog(Path("logs"))
write = blob.get_writer("sensor_data")
write(b"raw bytes")  # Timestamp added automatically
await blob.close()

# Read
blob = BlobLog(Path("logs"))
async for timestamp, data in blob.read_channel("sensor_data"):
    print(f"{timestamp}: {len(data)} bytes")
await blob.close()
```

**Key features:**
- Zero-copy memory-mapped reads
- Batched writes with `writev()`
- Timestamp-ordered merge of multiple logs
- ~150 lines, no dependencies

---

### 2. ObLog - Object Logging

Adds codec system on top of BlobLog for automatic encoding/decoding of messages.

```python
from pathlib import Path
from tinman.oblog import Codec, ObLog
import heapq
from collections import Counter

class HuffmanCodec(Codec[str]):
    """Compress strings using Huffman encoding based on training text."""
    
    def __init__(self, training_text: str):
        ...

    def encode(self, item: str) -> bytes:
        # Convert string to bits using encoding table, pack to bytes
        ...
    
    def decode(self, data: bytes) -> str:
        # Unpack bytes to bits, traverse tree to decode
        ...

# Write: codec parameters (Huffman tree) stored as first record
oblog = ObLog(Path("logs"))
write = oblog.get_writer("messages", HuffmanCodec("the quick brown fox..."))
write("hello world")  # Compressed using codec's tree
await oblog.close()

# Read: codec auto-restored from first record with exact same tree
oblog = ObLog(Path("logs"))
async for timestamp, text in oblog.read_channel("messages"):
    print(text)  # Decompressed correctly - "hello world"
await oblog.close()
```

- On read, codec is unpickled (with security restrictions) and used to decode remaining records
- Makes log files self-describing

**PickleCodec for prototyping:**
```python
from tinman import enable_pickle_codec

# PickleCodec not registered by default (security risk)
enable_pickle_codec()  # Now you can use it

# Without explicit codec, outputs use PickleCodec by default
async def producer(out: Annotated[Out[dict], "data"]):
    await out.publish({"any": "object"})  # Works with PickleCodec
```

**Built-in Codecs:**
```python
from tinman.codecs import NumpyArrayCodec
import numpy as np

# Zero-copy NumPy arrays (3x faster than pickle!)
oblog = ObLog(Path("logs"))
write = oblog.get_writer("sensor_data", NumpyArrayCodec())
write(np.random.rand(1000, 1000))  # 7.6 MB array
await oblog.close()

# Read with zero-copy (arrays are views into mmap)
async for timestamp, arr in oblog.read_channel("sensor_data"):
    print(arr.mean())  # No copy! Array is a view
```

See [codecs README](src/tinman/codecs/README.md) for more details and benchmarks.

**Key features:**
- Self-describing log files (codec in first record)
- Codec registry with auto-registration
- Safe unpickling with allowlist (only registered codecs)
- Optional `PickleCodec` for prototyping (must be explicitly enabled)
- Zero-copy NumPy codec (3x faster reads)
- ~200 lines

---

### 3. PubSub - Async Channels

Simple async pub/sub with backpressure.

```python
from tinman.pubsub import Out, In

out = Out[str]()
in1 = out.sub()
in2 = out.sub()

await out.publish("hello")  # Broadcasts to all subscribers

async for msg in in1:
    print(msg)
```

**Key features:**
- Generic types: `Out[T]` and `In[T]`
- Automatic close propagation
- Bounded queues for backpressure (configurable, default: 10)
- ~70 lines

**Configurable queue sizes:**
```python
# Large queue to buffer bursts
async def bursty_consumer(inp: Annotated[In[str], "messages", 100]):
    async for msg in inp:
        quick_operation(msg)
```

---

### 4. Runtime - Core Execution Engine

Low-level graph execution with automatic channel wiring and validation.

```python
from typing import Annotated
from tinman import In, Out, run
from tinman.runtime import get_node_specs, validate_nodes, run_nodes

async def producer(out: Annotated[Out[str], "messages"]):
    for i in range(10):
        await out.publish(f"message {i}")

async def consumer(inp: Annotated[In[str], "messages"]):
    async for msg in inp:
        print(msg)

# High-level: simple execution
await run([producer, consumer])

# Low-level: manual control
specs = get_node_specs([producer, consumer])
validate_nodes(specs)
await run_nodes(specs)
```

**Key features:**
- `get_node_specs()` - Extract channel metadata from node signatures
- `validate_nodes()` - Check graph connectivity (no duplicate outputs, all inputs have producers)
- `run_nodes()` - Wire channels and execute nodes concurrently
- `NodeSpec` - Dataclass for node metadata (inputs, outputs, dict injection)
- Introspection via `Annotated[In[T], "channel"]` and `Annotated[Out[T], "channel", codec]`
- Support for `dict[str, In]` to receive all channels (for logging/monitoring)
- ~370 lines

---

### 5. Launcher - High-Level Utilities

Convenient functions for common patterns: logging and playback.

```python
from typing import Annotated
from pathlib import Path
from tinman import In, Out, run, playback

async def sensors(out: Annotated[Out[dict], "raw_sensors"]):
    """Read sensor data at 10Hz"""
    for i in range(50):
        await out.publish({"lidar": 30 + i, "battery": 100 - i})
        await asyncio.sleep(0.1)

async def perception(
    inp: Annotated[In[dict], "raw_sensors"],
    out: Annotated[Out[dict], "world_state"]
):
    """Process sensors into world state"""
    async for sensors in inp:
        await out.publish({
            "obstacle": sensors["lidar"] < 30,
            "low_battery": sensors["battery"] < 25
        })

async def planner(
    inp: Annotated[In[dict], "world_state"],
    out: Annotated[Out[str], "commands"]
):
    """Make decisions"""
    async for world in inp:
        if world["obstacle"]:
            await out.publish("STOP")
        elif world["low_battery"]:
            await out.publish("DOCK")
        else:
            await out.publish("FORWARD")

# First run: Automatic logging with log_dir parameter
await run([sensors, perception, planner], log_dir=Path("logs"))

# Second run: Test new planner with recorded data
async def aggressive_planner(
    inp: Annotated[In[dict], "world_state"],
    out: Annotated[Out[str], "commands"]
):
    """Alternate decision logic"""
    async for world in inp:
        await out.publish("YOLO" if not world["obstacle"] else "STOP")

# Automatic playback - replays world_state, runs new planner
await playback([aggressive_planner], Path("logs"))

# Optional: control playback speed
await playback([aggressive_planner], Path("logs"), speed=1.0)
# speed=float('inf') (default): as fast as possible
# speed=1.0: realtime (respects original timestamps)
# speed=2.0: double speed
# speed=0.5: half speed
```

**Advanced usage (more control):**
```python
from tinman import get_node_specs
from tinman.launcher import create_logging_node, create_playback_graph

# Manual logging with channel filtering
specs = get_node_specs([sensors, perception, planner])
codecs = {ch: codec for spec in specs for _, (ch, codec) in spec.outputs.items()}
logger = create_logging_node(Path("logs"), codecs, channel_filter={"important_channel"})
await run([sensors, perception, planner, logger])

# Manual playback graph creation
graph = await create_playback_graph([aggressive_planner], Path("logs"))
await run(graph)
```

**Key features:**
- `run()` - Execute nodes, optionally with automatic logging via `log_dir=`
- `playback()` - Execute nodes with automatic playback from logs
- `create_logging_node()` - Factory for custom logging nodes with channel filtering
- `create_playback_graph()` - Factory for custom playback graphs
- Playback speed control
- ~300 lines
