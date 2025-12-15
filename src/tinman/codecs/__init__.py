"""Codecs for serializing objects to/from bytes.

This module provides various codec implementations for use with ObLog.

Recommended codecs:
- NumpyArrayCodec: Zero-copy for numpy arrays (122x faster than pickle!)
- DataFrameCodec: Zero-copy for DataFrames (memory-efficient, ~20-180μs decode)
- DataFrameParquetCodec: Compressed storage for DataFrames
"""

from .numpy import NumpyArrayCodec
from .pandas import DataFrameCodec, DataFrameParquetCodec

__all__ = [
    "NumpyArrayCodec",
    "DataFrameCodec",
    "DataFrameParquetCodec",
]
