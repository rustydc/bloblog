# Tinman

**A minimal async framework for building data pipelines with logging and playback.**

Tinman is built from four composable modules:

1. **BlobLog** - Fast binary logging (timestamp + raw bytes)
2. **ObLog** - Object logging with codecs (builds on BlobLog)
3. **PubSub** - Simple async pub/sub channels  
4. **Runner** - Autowired coroutines with logging and playback (uses all of the above)

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

**Key features:**
- Self-describing log files (codec in first record)
- Codec registry with auto-registration
- Safe unpickling with allowlist (only registered codecs)
- Optional `PickleCodec` for prototyping (must be explicitly enabled)
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

### 4. Runner - Autowired Coroutines

Wires async functions together with PubSub using type annotations, with automatic ObLog logging and playback.

```python
from typing import Annotated
from pathlib import Path
from tinman import In, Out, run, enable_pickle_codec

# Use pickle for prototyping (trusted data only)
enable_pickle_codec()

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

# First run: full pipeline with logging
await run([sensors, perception, planner], log_dir=Path("logs"))

# Second run: test new planner with recorded sensor data
async def aggressive_planner(
    inp: Annotated[In[dict], "world_state"],
    out: Annotated[Out[str], "commands"]
):
    """Alternate decision logic"""
    async for world in inp:
        await out.publish("YOLO" if not world["obstacle"] else "STOP")

await run([aggressive_planner], playback_dir=Path("logs"))
# This replays world_state from logs, runs new planner
# No need to re-run sensors or perception

# Optional: control playback speed
await run([aggressive_planner], playback_dir=Path("logs"), playback_speed=1.0)
# speed=0 (default): as fast as possible
# speed=1.0: realtime (respects original timestamps)
# speed=2.0: double speed
# speed=0.5: half speed
```

**Key features:**
- Automatic channel wiring from annotations
- Concurrent node execution
- Optional logging to all output channels
- Playback mode for testing/development
- Playback speed control (0=fast, 1.0=realtime, 2.0=2x, etc.)
- Validation (no duplicate outputs, all inputs have sources)
- ~350 lines
