"""Codecs for serializing objects to/from bytes.

This module provides various codec implementations for use with ObLog.

Recommended codecs:
- NumpyArrayCodec: Zero-copy for numpy arrays (122x faster than pickle!)
- DataFrameCodec: Zero-copy for DataFrames (memory-efficient, ~20-180μs decode)
- DataFrameParquetCodec: Compressed storage for DataFrames
- GeoDataFrameCodec: GeoParquet for GeoPandas GeoDataFrames

Composite codec utilities:
- DictCodec: Dicts with typed values (preserves zero-copy for array values)
- ListCodec: Lists with uniform element type
- TupleCodec: Tuples with heterogeneous types
- OptionalCodec: Wrap any codec to support None values

Primitive codecs (for building composites):
- IntCodec, FloatCodec, StringCodec, BoolCodec
"""

from .composite import (
    BoolCodec,
    DataclassCodec,
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    OptionalCodec,
    StringCodec,
    TupleCodec,
)
from .numpy import NumpyArrayCodec
from .pandas import DataFrameCodec, DataFrameParquetCodec
from .protobuf import ProtobufCodec
from .image import Image, ImageCodec

# Optional: GeoPandas codec (requires geopandas)
try:
    from .geopandas import GeoDataFrameCodec
except ImportError:
    pass

__all__ = [
    # Zero-copy specialized codecs
    "NumpyArrayCodec",
    "DataFrameCodec",
    "DataFrameParquetCodec",
    # Image codec for video/camera data
    "Image",
    "ImageCodec",
    # GeoPandas codec (optional)
    "GeoDataFrameCodec",
    # Protocol Buffer codec
    "ProtobufCodec",
    # Composite codec utilities
    "DataclassCodec",
    "DictCodec",
    "ListCodec",
    "TupleCodec",
    "OptionalCodec",
    # Primitive codecs
    "IntCodec",
    "FloatCodec",
    "StringCodec",
    "BoolCodec",
]
