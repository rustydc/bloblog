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

### Performance

See [benchmarks/NUMPY_CODEC_BENCHMARKS.md](../../benchmarks/NUMPY_CODEC_BENCHMARKS.md) for detailed benchmarks.

Summary:
- **3x faster** reads than pickle
- **5.7x faster** reads than writes (for large arrays)
- **0 MB** memory overhead for 76 MB arrays

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

## Future Codecs

Other codecs that could be added:
- `PandasDataFrameCodec` - Zero-copy DataFrames
- `ArrowCodec` - Apache Arrow format
- `ImageCodec` - Compressed images (PNG, JPEG)
- `ProtobufCodec` - Protocol buffers
- `JsonCodec` - JSON with schema
