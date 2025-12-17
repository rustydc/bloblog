"""Pure decode benchmarks - isolates zero-copy benefit without I/O overhead.

These benchmarks measure ONLY the decode operation to show the true
performance difference between zero-copy and copying codecs.
"""

import pickle
import struct
import numpy as np
import pytest

from tinman.codecs import NumpyArrayCodec


class PickleNumpyCodec:
    """Baseline: pickle-based codec (always copies)."""

    def encode(self, item: np.ndarray) -> bytes:
        return pickle.dumps(item)

    def decode(self, data: bytes) -> np.ndarray:
        return pickle.loads(data)


class NumpyCopyCodec:
    """Baseline: Same format as NumpyArrayCodec but forces a copy."""

    def encode(self, item: np.ndarray) -> bytes:
        dtype_str = item.dtype.str.encode("ascii")
        dtype_len = len(dtype_str)
        ndim = item.ndim
        
        header = struct.pack("<I", dtype_len) + dtype_str
        header += struct.pack("<I", ndim)
        header += struct.pack(f"<{ndim}Q", *item.shape)
        
        return header + item.tobytes()

    def decode(self, data: bytes) -> np.ndarray:
        """Same logic but forces copy by using bytes instead of memoryview."""
        offset = 0
        
        # Read dtype
        dtype_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dtype_str = data[offset : offset + dtype_len].decode("ascii")
        offset += dtype_len
        dtype = np.dtype(dtype_str)
        
        # Read shape
        ndim = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        shape = struct.unpack_from(f"<{ndim}Q", data, offset)
        offset += 8 * ndim
        
        # Force copy by slicing bytes and using frombuffer with copy
        array_data = data[offset:]
        arr = np.frombuffer(array_data, dtype=dtype).copy().reshape(shape)
        
        return arr


# Small arrays (1KB)
SMALL_ARRAY = np.random.rand(128).astype(np.float64)

# Medium arrays (1MB)  
MEDIUM_ARRAY = np.random.rand(128, 1024).astype(np.float64)

# Large arrays (10MB)
LARGE_ARRAY = np.random.rand(1280, 1024).astype(np.float64)


@pytest.mark.benchmark(group="decode-small")
def test_decode_pickle_small(benchmark):
    """Pickle decode - 1KB array."""
    codec = PickleNumpyCodec()
    data = codec.encode(SMALL_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-small")
def test_decode_copy_small(benchmark):
    """Copy decode - 1KB array."""
    codec = NumpyCopyCodec()
    data = codec.encode(SMALL_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-small")
def test_decode_zerocopy_small(benchmark):
    """Zero-copy decode - 1KB array."""
    codec = NumpyArrayCodec()
    data = codec.encode(SMALL_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-medium")
def test_decode_pickle_medium(benchmark):
    """Pickle decode - 1MB array."""
    codec = PickleNumpyCodec()
    data = codec.encode(MEDIUM_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-medium")
def test_decode_copy_medium(benchmark):
    """Copy decode - 1MB array."""
    codec = NumpyCopyCodec()
    data = codec.encode(MEDIUM_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-medium")
def test_decode_zerocopy_medium(benchmark):
    """Zero-copy decode - 1MB array."""
    codec = NumpyArrayCodec()
    data = codec.encode(MEDIUM_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-large")
def test_decode_pickle_large(benchmark):
    """Pickle decode - 10MB array."""
    codec = PickleNumpyCodec()
    data = codec.encode(LARGE_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-large")
def test_decode_copy_large(benchmark):
    """Copy decode - 10MB array."""
    codec = NumpyCopyCodec()
    data = codec.encode(LARGE_ARRAY)
    benchmark(codec.decode, data)


@pytest.mark.benchmark(group="decode-large")
def test_decode_zerocopy_large(benchmark):
    """Zero-copy decode - 10MB array."""
    codec = NumpyArrayCodec()
    data = codec.encode(LARGE_ARRAY)
    benchmark(codec.decode, data)


if __name__ == "__main__":
    """Quick manual test to show the difference."""
    import time
    
    print("=" * 70)
    print("Pure Decode Benchmarks - Zero-Copy vs Copy")
    print("=" * 70)
    
    # Test with large array (10MB)
    print(f"\nArray size: {LARGE_ARRAY.nbytes / 1024 / 1024:.1f} MB")
    
    # Pickle
    codec_pickle = PickleNumpyCodec()
    data_pickle = codec_pickle.encode(LARGE_ARRAY)
    t0 = time.perf_counter()
    for _ in range(100):
        arr = codec_pickle.decode(data_pickle)
    t_pickle = (time.perf_counter() - t0) / 100
    
    # Copy
    codec_copy = NumpyCopyCodec()
    data_copy = codec_copy.encode(LARGE_ARRAY)
    t0 = time.perf_counter()
    for _ in range(100):
        arr = codec_copy.decode(data_copy)
    t_copy = (time.perf_counter() - t0) / 100
    
    # Zero-copy
    codec_zero = NumpyArrayCodec()
    data_zero = codec_zero.encode(LARGE_ARRAY)
    t0 = time.perf_counter()
    for _ in range(100):
        arr = codec_zero.decode(data_zero)
    t_zero = (time.perf_counter() - t0) / 100
    
    print(f"\nPickle decode:    {t_pickle*1000:.2f} ms ({t_pickle/t_zero:.1f}x slower)")
    print(f"Copy decode:      {t_copy*1000:.2f} ms ({t_copy/t_zero:.1f}x slower)")
    print(f"Zero-copy decode: {t_zero*1000:.2f} ms (baseline)")
    
    print(f"\n✓ Zero-copy is {t_pickle/t_zero:.1f}x faster than pickle!")
    print(f"✓ Zero-copy is {t_copy/t_zero:.1f}x faster than copying!")
