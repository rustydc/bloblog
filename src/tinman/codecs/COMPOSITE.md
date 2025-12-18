# Composite Codecs

The `tinman.codecs.composite` module provides utilities for building codecs that handle nested data structures while preserving zero-copy semantics for NumPy arrays and other zero-copy types.

## Overview

Composite codecs let you serialize complex nested structures like:
- **Dicts** with typed values (e.g., `{"array": np.ndarray, "label": str}`)
- **Lists** of uniform types (e.g., list of arrays)
- **Tuples** with heterogeneous types (e.g., `(array, str, float)`)
- **Optional** fields that can be `None`

**Key benefit**: When you nest zero-copy codecs (like `NumpyArrayCodec`) inside composite structures, the arrays remain zero-copy views into the memory-mapped file!

## Available Codecs

### DataclassCodec

For type-safe dataclass instances (recommended for structured data):

```python
from dataclasses import dataclass
from tinman.codecs import DataclassCodec, NumpyArrayCodec, StringCodec, IntCodec

@dataclass
class SensorReading:
    data: np.ndarray
    label: str
    confidence: int

# Define schema (same as DictCodec)
schema = {
    'data': NumpyArrayCodec(),
    'label': StringCodec(),
    'confidence': IntCodec(),
}
codec = DataclassCodec(SensorReading, schema)

# Write dataclass instances
reading = SensorReading(
    data=np.array([1, 2, 3]),
    label='sensor_a',
    confidence=95
)
write(reading)

# Read back as type-safe dataclass instances
reader = ObLogReader(Path("logs"))
async for _, reading in reader.read_channel("sensors"):
    print(reading.data)        # Attribute access (IDE autocomplete!)
    print(reading.label)       # Type checker validates
    print(reading.confidence)  # Self-documenting
```

**Benefits:**
- ✅ Type-safe attribute access
- ✅ IDE autocomplete and refactoring support
- ✅ Type checker validates field usage
- ✅ Self-documenting with type hints
- ✅ Same efficient wire format as DictCodec
- ✅ Zero-copy arrays still work!

### DictCodec

For dicts with typed values:

```python
from tinman.codecs import DictCodec, NumpyArrayCodec, StringCodec, IntCodec

# Define schema
schema = {
    "data": NumpyArrayCodec(),
    "label": StringCodec(),
    "count": IntCodec(),
}
codec = DictCodec(schema)

# Use it
data = {
    "data": np.array([1, 2, 3]),
    "label": "sensor_a",
    "count": 42,
}
write(data)
```

### ListCodec

For lists where all elements have the same type:

```python
from tinman.codecs import ListCodec, NumpyArrayCodec

# List of arrays (like a batch)
codec = ListCodec(NumpyArrayCodec())

# Use it
batch = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
]
write(batch)
```

### TupleCodec

For tuples with heterogeneous types:

```python
from tinman.codecs import TupleCodec, NumpyArrayCodec, StringCodec, FloatCodec

# Tuple: (array, label, confidence)
codec = TupleCodec([
    NumpyArrayCodec(),
    StringCodec(),
    FloatCodec(),
])

# Use it
write((np.array([1, 2, 3]), "cat", 0.95))
```

### OptionalCodec

Wraps another codec to support `None` values:

```python
from tinman.codecs import OptionalCodec, StringCodec, DictCodec

# Dict with optional error field
schema = {
    "success": IntCodec(),
    "error": OptionalCodec(StringCodec()),
}
codec = DictCodec(schema)

# Use it
write({"success": 1, "error": None})
write({"success": 0, "error": "Timeout"})
```

**Alternative**: For dict keys, you can use `allow_missing=True` instead:

```python
# Simpler approach for missing dict keys
schema = {
    "success": IntCodec(),
    "error": StringCodec(),
}
codec = DictCodec(schema, allow_missing=True)

# Missing keys are simply not present in the dict
write({"success": 1})  # error key missing - encoded as length=0
write({"success": 0, "error": "Timeout"})  # error key present

# On decode, missing keys are not in the result dict
reader = ObLogReader(Path("logs"))
async for _, record in reader.read_channel("results"):
    if 'error' in record:
        print(record['error'])  # Only present if it was written
```

## Primitive Codecs

For building composite structures:

- **IntCodec**: 64-bit signed integers
- **FloatCodec**: 64-bit floats
- **StringCodec**: UTF-8 strings
- **BoolCodec**: Booleans

```python
from tinman.codecs import IntCodec, FloatCodec, StringCodec, BoolCodec
```

## Nested Structures

You can nest composite codecs arbitrarily:

```python
from tinman.codecs import DictCodec, ListCodec, NumpyArrayCodec, IntCodec

# Dict containing lists of arrays
schema = {
    "train_batch": ListCodec(NumpyArrayCodec()),
    "val_batch": ListCodec(NumpyArrayCodec()),
    "epoch": IntCodec(),
}
codec = DictCodec(schema)

# Use it
data = {
    "train_batch": [np.random.rand(10, 5) for _ in range(3)],
    "val_batch": [np.random.rand(10, 5) for _ in range(2)],
    "epoch": 0,
}
write(data)
```

## Zero-Copy Guarantee

**Critical property**: When you read nested structures, any arrays decoded with `NumpyArrayCodec` are still zero-copy views into the memory-mapped file!

```python
# After reading
reader = ObLogReader(Path("logs"))
async for _, data in reader.read_channel("training"):
    # data['train_batch'] is a list
    # Each array in the list is a zero-copy view!
    for arr in data['train_batch']:
        assert not arr.flags.writeable  # Read-only view into mmap
```

This means:
- ✅ No memory allocation for array data
- ✅ Multiple reads share the same memory
- ✅ Excellent memory efficiency
- ✅ Fast deserialization

## Complete Example

```python
import asyncio
from pathlib import Path
import numpy as np
from tinman import ObLogWriter, ObLogReader
from tinman.codecs import (
    DictCodec,
    ListCodec,
    NumpyArrayCodec,
    StringCodec,
    IntCodec,
    FloatCodec,
)

async def main():
    # Define schema for sensor readings
    schema = {
        "accelerometer": NumpyArrayCodec(),
        "gyroscope": NumpyArrayCodec(),
        "timestamp": IntCodec(),
        "device_id": StringCodec(),
        "temperature": FloatCodec(),
    }
    codec = DictCodec(schema)
    
    # Write
    async with ObLogWriter(Path("logs")) as oblog:
        write = oblog.get_writer("sensors", codec)
        
        write({
            "accelerometer": np.random.rand(100, 3),
            "gyroscope": np.random.rand(100, 3),
            "timestamp": 1000000,
            "device_id": "sensor_0",
            "temperature": 25.3,
        })
    
    # Read (zero-copy for arrays!)
    reader = ObLogReader(Path("logs"))
    async for timestamp, reading in reader.read_channel("sensors"):
        print(f"Device {reading['device_id']}")
        print(f"  Accel: {reading['accelerometer'].shape}")
        print(f"  Gyro: {reading['gyroscope'].shape}")
        # Arrays are read-only views - no memory allocated!
        assert not reading['accelerometer'].flags.writeable

asyncio.run(main())
```

## When to Use

### Use Composite Codecs When:
- ✅ You have structured data with multiple fields
- ✅ You want type safety and schema definition
- ✅ You want zero-copy for nested arrays/dataframes
- ✅ You're building reusable data formats

### Use PickleCodec When:
- ✅ Prototyping/quick experiments
- ✅ Data structure changes frequently
- ✅ Don't need maximum performance
- ⚠️ You trust the data source (pickle has security issues)

### Use Parquet/Arrow When:
- ✅ Need cross-language compatibility
- ✅ Need compression
- ✅ Working with tabular data
- ❌ Don't need zero-copy reads (they decompress)

## See Also

- [numpy.py](numpy.py) - Zero-copy NumPy array codec
- [pandas.py](pandas.py) - Zero-copy DataFrame codec
- [README.md](README.md) - Main codecs documentation
- [examples/composite_codecs.py](../../examples/composite_codecs.py) - Full examples
