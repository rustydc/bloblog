"""NumPy array codec with zero-copy reads.

This codec stores numpy arrays in a format that allows zero-copy reading
when data is memory-mapped from disk.
"""

from __future__ import annotations

import struct
from collections.abc import Buffer

import numpy as np

from ..oblog import Codec


class NumpyArrayCodec(Codec[np.ndarray]):
    """Zero-copy codec for numpy arrays.
    
    Stores dtype and shape metadata, then raw array data.
    On decode, creates numpy array directly from memoryview without copying.
    
    Format:
        - dtype_len: uint32 (4 bytes)
        - dtype_str: ASCII string
        - ndim: uint32 (4 bytes)
        - shape: ndim × uint64 (8 bytes each)
        - data: raw array bytes
    
    Example:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> from tinman import ObLog
        >>> from tinman.codecs import NumpyArrayCodec
        >>> 
        >>> # Write arrays
        >>> oblog = ObLog(Path("logs"))
        >>> write = oblog.get_writer("arrays", NumpyArrayCodec())
        >>> write(np.array([1, 2, 3], dtype=np.int32))
        >>> await oblog.close()
        >>> 
        >>> # Read arrays (zero-copy)
        >>> oblog = ObLog(Path("logs"))
        >>> async for timestamp, arr in oblog.read_channel("arrays"):
        >>>     print(arr)  # numpy array view into mmap
        >>> await oblog.close()
    
    Note:
        The returned arrays are read-only views into the memory-mapped file.
        As long as you keep a reference to the array, the underlying mmap
        stays open automatically via weakref finalizers.
    """

    def encode(self, item: np.ndarray) -> bytes:
        """Encode numpy array to bytes.
        
        Args:
            item: NumPy array to encode
            
        Returns:
            Bytes containing metadata and array data
        """
        # Store: dtype string length (4 bytes), dtype string, ndim (4 bytes), shape, data
        dtype_str = item.dtype.str.encode("ascii")
        dtype_len = len(dtype_str)
        ndim = item.ndim
        
        header = struct.pack("<I", dtype_len) + dtype_str
        header += struct.pack("<I", ndim)
        header += struct.pack(f"<{ndim}Q", *item.shape)
        
        # Return header + raw data bytes
        return header + item.tobytes()

    def decode(self, data: Buffer) -> np.ndarray:
        """Decode buffer/memoryview to numpy array (zero-copy).
        
        Args:
            data: Buffer/memoryview containing encoded array
            
        Returns:
            NumPy array that is a VIEW into the memoryview.
            The array is read-only to prevent modification of the mmap.
            
        Note:
            The returned array is a zero-copy view into the source data.
            As long as you keep a reference to the array, the underlying
            memory-mapped file stays open automatically via weakref finalizers.
        """
        # Convert to memoryview for slicing
        mv = memoryview(data)
        offset = 0
        
        # Read dtype
        dtype_len = struct.unpack_from("<I", mv, offset)[0]
        offset += 4
        dtype_str = bytes(mv[offset : offset + dtype_len]).decode("ascii")
        offset += dtype_len
        dtype = np.dtype(dtype_str)
        
        # Read shape
        ndim = struct.unpack_from("<I", mv, offset)[0]
        offset += 4
        shape = struct.unpack_from(f"<{ndim}Q", mv, offset)
        offset += 8 * ndim
        
        # Create numpy array from memoryview (zero-copy!)
        # The memoryview keeps the mmap alive via weakref finalizer
        array_data = mv[offset:]
        arr = np.frombuffer(array_data, dtype=dtype).reshape(shape)
        
        # Make it read-only to prevent accidental modification of mmap
        arr.flags.writeable = False
        
        return arr
