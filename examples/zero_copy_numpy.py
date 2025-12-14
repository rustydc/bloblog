"""Example: Zero-copy numpy and pandas integration.

This demonstrates how to use BlobLog's memoryview slices directly with
numpy arrays and pandas DataFrames without copying data.

Run with: uv run python -m examples.zero_copy_numpy
"""

import asyncio
import struct
from collections.abc import Buffer
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from tinman import ObLog
from tinman.codecs import NumpyArrayCodec
from tinman.oblog import Codec


class DataFrameCodec(Codec[pd.DataFrame]):
    """Zero-copy codec for pandas DataFrames.
    
    Stores column names and dtypes, then uses record array format.
    Works best for numeric columns that can be stored contiguously.
    """

    def encode(self, item: pd.DataFrame) -> bytes:
        """Encode DataFrame to bytes."""
        # Store: number of columns, column names and dtypes, then data
        cols = item.columns.tolist()
        n_cols = len(cols)
        
        # Serialize column metadata
        col_metadata = []
        for col in cols:
            col_name = col.encode("utf-8")
            dtype_str = str(item[col].dtype).encode("utf-8")
            col_metadata.append(
                struct.pack("<I", len(col_name))
                + col_name
                + struct.pack("<I", len(dtype_str))
                + dtype_str
            )
        
        header = struct.pack("<I", n_cols) + b"".join(col_metadata)
        
        # Convert to numpy record array and get bytes
        # This is most efficient for numeric data
        records = item.to_records(index=False)
        return header + records.tobytes()

    def decode(self, data: Buffer) -> pd.DataFrame:
        """Decode buffer/memoryview to DataFrame (zero-copy for numeric columns).
        
        The DataFrame's underlying arrays are views into the memoryview.
        Keep a reference to the DataFrame to keep the mmap alive.
        """
        # Convert to memoryview for slicing
        mv = memoryview(data)
        offset = 0
        
        # Read column metadata
        n_cols = struct.unpack_from("<I", mv, offset)[0]
        offset += 4
        
        columns = []
        dtypes = []
        for _ in range(n_cols):
            name_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            col_name = bytes(mv[offset : offset + name_len]).decode("utf-8")
            offset += name_len
            
            dtype_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            dtype_str = bytes(mv[offset : offset + dtype_len]).decode("utf-8")
            offset += dtype_len
            
            columns.append(col_name)
            dtypes.append((col_name, dtype_str))
        
        # Create structured dtype
        dtype = np.dtype(dtypes)
        
        # Create numpy structured array from memoryview (zero-copy!)
        array_data = mv[offset:]
        records = np.frombuffer(array_data, dtype=dtype)
        records.flags.writeable = False
        
        # Convert to DataFrame (this creates views, not copies)
        return pd.DataFrame(records)


async def demo_numpy_arrays():
    """Demonstrate zero-copy numpy array logging."""
    print("\n=== NumPy Array Demo ===\n")
    
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Write some arrays
        print("Writing arrays...")
        oblog = ObLog(log_dir)
        write = oblog.get_writer("arrays", NumpyArrayCodec())
        
        # Write different shaped arrays
        arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.random.rand(100, 50).astype(np.float32),  # Larger array
        ]
        
        for i, arr in enumerate(arrays):
            write(arr)
            print(f"  Wrote array {i}: shape={arr.shape}, dtype={arr.dtype}")
        
        await oblog.close()
        
        # Read back with zero-copy
        print("\nReading arrays (zero-copy)...")
        oblog = ObLog(log_dir)
        
        read_arrays = []
        async for timestamp, arr in oblog.read_channel("arrays"):
            print(f"  Read array: shape={arr.shape}, dtype={arr.dtype}, writeable={arr.flags.writeable}")
            
            # Verify it's truly zero-copy by checking the base
            print(f"    Is view: {arr.base is not None}")
            
            # You can do computations on the array
            print(f"    Sum: {arr.sum()}")
            
            # Keep reference to prevent mmap from closing
            read_arrays.append(arr)
        
        await oblog.close()
        
        # Verify data integrity
        print("\nVerifying data...")
        for orig, read in zip(arrays, read_arrays):
            assert np.array_equal(orig, read), "Data mismatch!"
        print("  ✓ All arrays match!")


async def demo_dataframes():
    """Demonstrate zero-copy pandas DataFrame logging."""
    print("\n=== Pandas DataFrame Demo ===\n")
    
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Write some DataFrames
        print("Writing DataFrames...")
        oblog = ObLog(log_dir)
        write = oblog.get_writer("dataframes", DataFrameCodec())
        
        # Create sample DataFrames
        dfs = [
            pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}),
            pd.DataFrame(
                {
                    "x": np.random.rand(1000),
                    "y": np.random.rand(1000),
                    "z": np.random.randint(0, 100, 1000),
                }
            ),
        ]
        
        for i, df in enumerate(dfs):
            write(df)
            print(f"  Wrote DataFrame {i}: shape={df.shape}, size={df.memory_usage(deep=True).sum()} bytes")
        
        await oblog.close()
        
        # Read back with zero-copy
        print("\nReading DataFrames (zero-copy)...")
        oblog = ObLog(log_dir)
        
        read_dfs = []
        async for timestamp, df in oblog.read_channel("dataframes"):
            print(f"  Read DataFrame: shape={df.shape}")
            print(f"    Columns: {df.columns.tolist()}")
            print(f"    First row: {df.iloc[0].to_dict()}")
            
            # The underlying arrays are views
            for col in df.columns:
                arr = df[col].values
                print(f"    Column '{col}' is view: {arr.base is not None}")
            
            read_dfs.append(df)
        
        await oblog.close()
        
        # Verify data integrity
        print("\nVerifying data...")
        for orig, read in zip(dfs, read_dfs):
            pd.testing.assert_frame_equal(orig, read)
        print("  ✓ All DataFrames match!")


async def demo_zero_copy_performance():
    """Show that zero-copy reads don't load entire file into memory."""
    print("\n=== Zero-Copy Performance Demo ===\n")
    
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Write a large array
        print("Writing large array (100MB)...")
        oblog = ObLog(log_dir)
        write = oblog.get_writer("large", NumpyArrayCodec())
        
        large_array = np.random.rand(1000, 10000).astype(np.float64)
        size_mb = large_array.nbytes / 1024 / 1024
        print(f"  Array size: {size_mb:.1f} MB")
        write(large_array)
        await oblog.close()
        
        # Read it back - the mmap is created but data isn't loaded
        print("\nReading with zero-copy (mmap)...")
        oblog = ObLog(log_dir)
        
        async for timestamp, arr in oblog.read_channel("large"):
            print(f"  Got array reference: shape={arr.shape}")
            print(f"  Array is view: {arr.base is not None}")
            
            # Access just one element - only that page is loaded
            print(f"  Accessing single element: arr[0, 0] = {arr[0, 0]}")
            
            # Compute on a slice - only needed pages are loaded
            print(f"  Computing sum of first 10 rows: {arr[:10].sum()}")
            
            print("\n  ✓ Only accessed data is loaded into memory!")
            print("    The rest stays in the mmap until you touch it.")
        
        await oblog.close()


if __name__ == "__main__":
    asyncio.run(demo_numpy_arrays())
    asyncio.run(demo_dataframes())
    asyncio.run(demo_zero_copy_performance())
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("✓ BlobLog returns memoryview slices (zero-copy)")
    print("✓ NumPy can create arrays directly from memoryviews")
    print("✓ Pandas DataFrames can use numpy views as columns")
    print("✓ Data stays in mmap until accessed (lazy loading)")
    print("✓ weakref finalizers keep mmap alive as long as arrays exist")
    print("=" * 60)
