# Tinman Codecs

This module provides codecs for serializing objects to/from bytes for use with ObLog.

## NumpyArrayCodec

Zero-copy codec for NumPy arrays with memory-mapped reads.

### Features

- ✅ **Zero-copy reads** - Arrays are views into mmap'd files
- ✅ **3x faster** than pickle for reading
- ✅ **No memory overhead** for large arrays
- ✅ **Lazy loading** - Only accessed data is paged in
- ✅ **Safe** - Arrays are read-only by default
- ✅ **Universal** - Works with any dtype/shape

### Usage

```python
from pathlib import Path
from tinman import ObLog
from tinman.codecs import NumpyArrayCodec
import numpy as np

# Write arrays
oblog = ObLog(Path("logs"))
write = oblog.get_writer("sensor_data", NumpyArrayCodec())

for i in range(100):
    data = np.random.rand(1000)
    write(data)

await oblog.close()

# Read arrays (zero-copy!)
oblog = ObLog(Path("logs"))
async for timestamp, arr in oblog.read_channel("sensor_data"):
    # arr is a numpy array, but it's a VIEW into the mmap
    print(f"{timestamp}: {arr.mean()}")
    
    # Verify zero-copy
    print(f"Is view: {arr.base is not None}")  # True
    print(f"Read-only: {not arr.flags.writeable}")  # True

await oblog.close()
```

### Format

The codec stores arrays in a compact binary format:

```
┌──────────────┬─────────────┬──────┬─────────┬──────────┐
│ dtype_len    │ dtype_str   │ ndim │ shape   │ data     │
│ (uint32)     │ (ASCII)     │ (u32)│ (u64[]) │ (bytes)  │
└──────────────┴─────────────┴──────┴─────────┴──────────┘
     4 bytes      variable     4 B   8*ndim B   variable
```

### Supported Dtypes

All NumPy dtypes are supported:
- Integers: `int8`, `int16`, `int32`, `int64`, `uint8`, etc.
- Floats: `float16`, `float32`, `float64`
- Complex: `complex64`, `complex128`
- Bool: `bool_`
- And more!

### Implementation Details

1. **Encoding**: Array metadata (dtype, shape) + raw bytes
2. **Decoding**: 
   - File is memory-mapped
   - `np.frombuffer(memoryview)` creates array directly from mmap
   - Array is marked read-only to prevent corruption
3. **Lifetime**: weakref finalizers keep mmap open as long as arrays exist
4. **Safety**: Arrays can't accidentally modify the underlying file

### Example: Time-Series Analysis

```python
# Log sensor data
oblog = ObLog(Path("sensor_logs"))
write = oblog.get_writer("temperature", NumpyArrayCodec())

for reading in sensor_readings:
    write(np.array(reading))

await oblog.close()

# Analyze later (zero-copy!)
oblog = ObLog(Path("sensor_logs"))
all_temps = []
async for _, temps in oblog.read_channel("temperature"):
    all_temps.append(temps)

# Concatenate into single array
temperatures = np.concatenate(all_temps)
print(f"Mean: {temperatures.mean()}")
print(f"Max: {temperatures.max()}")
```

### Limitations

1. **Read-only**: Decoded arrays are immutable (by design)
2. **Python types only**: Use `PickleCodec` for arbitrary Python objects
3. **Memory-mapped**: Requires file system support for mmap

If you need to modify arrays, make a copy:
```python
async for _, arr in oblog.read_channel("data"):
    mutable_arr = arr.copy()  # Now you can modify it
    mutable_arr[0] = 42
```

## DataFrameCodec

Zero-copy codec for pandas DataFrames with memory-mapped reads.

### Features

- ✅ **Zero-copy reads** for numeric columns
- ✅ **2.3x faster** than copying for large DataFrames
- ✅ **No memory overhead** - columns are views
- ✅ **Works with any numeric dtype**
- ✅ **Read-only** for safety

### Usage

```python
from pathlib import Path
from tinman import ObLog
from tinman.codecs import DataFrameCodec
import pandas as pd
import numpy as np

# Write sensor data
oblog = ObLog(Path("sensor_logs"))
write = oblog.get_writer("readings", DataFrameCodec())

df = pd.DataFrame({
    'timestamp': np.arange(1000, dtype=np.int64),
    'temperature': 20.0 + np.random.randn(1000) * 2,
    'humidity': 65.0 + np.random.randn(1000) * 5,
})
write(df)
await oblog.close()

# Read with zero-copy
oblog = ObLog(Path("sensor_logs"))
async for _, df in oblog.read_channel("readings"):
    print(f"Mean temp: {df['temperature'].mean()}")
    # Columns are views!
await oblog.close()
```

### DataFrameParquetCodec

Alternative codec using Apache Parquet format:
- Highly compressed
- Industry standard
- NOT zero-copy (decompression required)
- Best for long-term storage and compatibility

```python
from tinman.codecs import DataFrameParquetCodec

# Use Parquet for compressed storage
write = oblog.get_writer("archive", DataFrameParquetCodec())
```
