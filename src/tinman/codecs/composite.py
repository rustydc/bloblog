"""Utilities for building composite codecs that combine multiple codecs.

This module provides helpers to easily create codecs for nested structures like:
- Dicts containing arrays/dataframes
- Lists of arrays
- Tuples of mixed types
- Custom nested structures
"""

from __future__ import annotations

import struct
from collections.abc import Buffer, Mapping
from dataclasses import is_dataclass
from typing import Any, Protocol, TypeVar

from ..oblog import Codec


class _DataclassInstance(Protocol):
    """Protocol for dataclass instances (for type checking)."""

    __dataclass_fields__: dict[str, Any]


# Type variable for dataclasses bound to dataclass protocol
T = TypeVar("T", bound=_DataclassInstance)


class DictCodec(Codec[dict[str, Any]]):
    """Codec for dicts with heterogeneous values, each with its own codec.
    
    Supports zero-copy for values decoded with zero-copy codecs (like NumpyArrayCodec).
    Missing keys are encoded as length=0 and decoded as absent.
    
    Example:
        >>> from tinman.codecs import NumpyArrayCodec, DictCodec, StringCodec
        >>> import numpy as np
        >>> 
        >>> # Define schema: key -> codec
        >>> schema = {
        ...     'sensor_1': NumpyArrayCodec(),
        ...     'sensor_2': NumpyArrayCodec(),
        ...     'metadata': StringCodec(),
        ... }
        >>> codec = DictCodec(schema)
        >>> 
        >>> # Write
        >>> data = {
        ...     'sensor_1': np.array([1, 2, 3]),
        ...     'sensor_2': np.array([4, 5, 6]),
        ...     'metadata': 'test'
        ... }
        >>> oblog.get_writer("data", codec)(data)
        >>> 
        >>> # Read - arrays are zero-copy views!
        >>> async for _, item in oblog.read_channel("data"):
        ...     print(item['sensor_1'])  # Zero-copy numpy array
    """

    def __init__(self, schema: Mapping[str, Codec], *, allow_missing: bool = False):
        """Initialize with a schema mapping keys to codecs.
        
        Args:
            schema: Dict/Mapping of key names to their codecs.
                    Keys must be strings.
        """
        self.schema = dict(schema)
        self.allow_missing = allow_missing
        # Store key order for consistent encoding
        self.keys = sorted(self.schema.keys())

    def encode(self, item: dict[str, Any]) -> bytes:
        """Encode dict to bytes using schema codecs.
        
        Args:
            item: Dict with keys matching the schema.
                  If allow_missing=True, keys can be absent.
            
        Returns:
            Bytes containing encoded dict data.
            
        Raises:
            KeyError: If allow_missing=False and item is missing a key from schema.
        """
        # Encode in sorted key order for consistency
        parts = []
        for key in self.keys:
            if key not in item:
                if not self.allow_missing:
                    raise KeyError(f"Key '{key}' from schema not found in item")
                # Missing key: encode as length=0
                parts.append(struct.pack("<I", 0))
            else:
                value = item[key]
                codec = self.schema[key]
                
                # Encode value
                value_bytes = codec.encode(value)
                
                # Store length + data
                parts.append(struct.pack("<I", len(value_bytes)))
                parts.append(value_bytes)
        
        return b"".join(parts)

    def decode(self, data: Buffer) -> dict[str, Any]:
        """Decode buffer to dict using schema codecs (zero-copy where possible).
        
        Args:
            data: Buffer/memoryview containing encoded dict.
            
        Returns:
            Dict with decoded values. If a codec supports zero-copy (like NumpyArrayCodec),
            the value will be a view into the buffer.
            If allow_missing=True and a key was missing (length=0), it won't be in the result.
        """
        mv = memoryview(data)
        offset = 0
        result = {}
        
        for key in self.keys:
            codec = self.schema[key]
            
            # Read length
            value_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            
            if value_len == 0:
                # Missing key - skip it (don't add to result dict)
                continue
            
            # Decode value (zero-copy if codec supports it)
            value_data = mv[offset : offset + value_len]
            result[key] = codec.decode(value_data)
            offset += value_len
        
        return result


class ListCodec(Codec[list[Any]]):
    """Codec for lists where all elements use the same codec.
    
    Supports zero-copy for elements decoded with zero-copy codecs.
    
    Example:
        >>> from tinman.codecs import NumpyArrayCodec, ListCodec
        >>> import numpy as np
        >>> 
        >>> # Codec for list of arrays
        >>> codec = ListCodec(NumpyArrayCodec())
        >>> 
        >>> # Write
        >>> data = [
        ...     np.array([1, 2, 3]),
        ...     np.array([4, 5, 6]),
        ...     np.array([7, 8, 9])
        ... ]
        >>> oblog.get_writer("arrays", codec)(data)
        >>> 
        >>> # Read - each array is zero-copy!
        >>> async for _, item_list in oblog.read_channel("arrays"):
        ...     for arr in item_list:
        ...         print(arr)  # Zero-copy numpy array
    """

    def __init__(self, element_codec: Codec):
        """Initialize with codec for list elements.
        
        Args:
            element_codec: Codec to use for all list elements.
        """
        self.element_codec = element_codec

    def encode(self, item: list[Any]) -> bytes:
        """Encode list to bytes.
        
        Args:
            item: List of items to encode.
            
        Returns:
            Bytes containing encoded list data.
        """
        parts = [struct.pack("<I", len(item))]
        
        for element in item:
            element_bytes = self.element_codec.encode(element)
            parts.append(struct.pack("<I", len(element_bytes)))
            parts.append(element_bytes)
        
        return b"".join(parts)

    def decode(self, data: Buffer) -> list[Any]:
        """Decode buffer to list (zero-copy where possible).
        
        Args:
            data: Buffer/memoryview containing encoded list.
            
        Returns:
            List of decoded elements. If element_codec supports zero-copy,
            elements will be views into the buffer.
        """
        mv = memoryview(data)
        offset = 0
        
        # Read length
        length = struct.unpack_from("<I", mv, offset)[0]
        offset += 4
        
        result = []
        for _ in range(length):
            # Read element
            element_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            
            element_data = mv[offset : offset + element_len]
            result.append(self.element_codec.decode(element_data))
            offset += element_len
        
        return result


class TupleCodec(Codec[tuple]):
    """Codec for tuples with heterogeneous elements, each with its own codec.
    
    Like DictCodec but for positional values instead of named keys.
    Supports zero-copy for elements decoded with zero-copy codecs.
    
    Example:
        >>> from tinman.codecs import NumpyArrayCodec, TupleCodec
        >>> import numpy as np
        >>> 
        >>> # Codec for (array, array, int) tuples
        >>> codec = TupleCodec([
        ...     NumpyArrayCodec(),
        ...     NumpyArrayCodec(),
        ...     IntCodec(),  # Your custom int codec
        ... ])
        >>> 
        >>> # Write
        >>> data = (np.array([1, 2]), np.array([3, 4]), 42)
        >>> oblog.get_writer("data", codec)(data)
        >>> 
        >>> # Read - arrays are zero-copy!
        >>> async for _, item in oblog.read_channel("data"):
        ...     arr1, arr2, count = item
        ...     print(arr1)  # Zero-copy numpy array
    """

    def __init__(self, element_codecs: list[Codec]):
        """Initialize with codecs for each tuple position.
        
        Args:
            element_codecs: List of codecs, one per tuple element.
        """
        self.element_codecs = element_codecs

    def encode(self, item: tuple) -> bytes:
        """Encode tuple to bytes.
        
        Args:
            item: Tuple to encode.
            
        Returns:
            Bytes containing encoded tuple data.
            
        Raises:
            ValueError: If tuple length doesn't match number of codecs.
        """
        if len(item) != len(self.element_codecs):
            raise ValueError(
                f"Tuple has {len(item)} elements but {len(self.element_codecs)} codecs provided"
            )
        
        parts = []
        for element, codec in zip(item, self.element_codecs):
            element_bytes = codec.encode(element)
            parts.append(struct.pack("<I", len(element_bytes)))
            parts.append(element_bytes)
        
        return b"".join(parts)

    def decode(self, data: Buffer) -> tuple:
        """Decode buffer to tuple (zero-copy where possible).
        
        Args:
            data: Buffer/memoryview containing encoded tuple.
            
        Returns:
            Tuple of decoded elements. If a codec supports zero-copy,
            that element will be a view into the buffer.
        """
        mv = memoryview(data)
        offset = 0
        
        result = []
        for codec in self.element_codecs:
            element_len = struct.unpack_from("<I", mv, offset)[0]
            offset += 4
            
            element_data = mv[offset : offset + element_len]
            result.append(codec.decode(element_data))
            offset += element_len
        
        return tuple(result)


class OptionalCodec(Codec[Any | None]):
    """Codec that wraps another codec to support None values.
    
    Useful for optional fields in composite structures.
    
    Example:
        >>> from tinman.codecs import NumpyArrayCodec, OptionalCodec, DictCodec
        >>> 
        >>> # Dict with optional array field
        >>> schema = {
        ...     'required': NumpyArrayCodec(),
        ...     'optional': OptionalCodec(NumpyArrayCodec()),
        ... }
        >>> codec = DictCodec(schema)
        >>> 
        >>> # Can write with or without the optional field
        >>> data1 = {'required': np.array([1, 2]), 'optional': np.array([3, 4])}
        >>> data2 = {'required': np.array([1, 2]), 'optional': None}
    """

    def __init__(self, inner_codec: Codec):
        """Initialize with codec for non-None values.
        
        Args:
            inner_codec: Codec to use when value is not None.
        """
        self.inner_codec = inner_codec

    def encode(self, item: Any | None) -> bytes:
        """Encode optional value to bytes.
        
        Args:
            item: Value to encode, or None.
            
        Returns:
            Bytes with is_none flag followed by encoded value if not None.
        """
        if item is None:
            return struct.pack("<B", 1)  # is_none = True
        else:
            inner_bytes = self.inner_codec.encode(item)
            return struct.pack("<B", 0) + inner_bytes  # is_none = False

    def decode(self, data: Buffer) -> Any | None:
        """Decode buffer to optional value (zero-copy if inner codec supports it).
        
        Args:
            data: Buffer/memoryview containing encoded optional value.
            
        Returns:
            Decoded value or None.
        """
        mv = memoryview(data)
        is_none = struct.unpack_from("<B", mv, 0)[0]
        
        if is_none:
            return None
        else:
            return self.inner_codec.decode(mv[1:])


# Simple primitive codecs for use in composite structures

class IntCodec(Codec[int]):
    """Simple codec for 64-bit signed integers."""

    def encode(self, item: int) -> bytes:
        return struct.pack("<q", item)

    def decode(self, data: Buffer) -> int:
        return struct.unpack_from("<q", data, 0)[0]


class FloatCodec(Codec[float]):
    """Simple codec for 64-bit floats."""

    def encode(self, item: float) -> bytes:
        return struct.pack("<d", item)

    def decode(self, data: Buffer) -> float:
        return struct.unpack_from("<d", data, 0)[0]


class StringCodec(Codec[str]):
    """Simple UTF-8 string codec."""

    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


class BoolCodec(Codec[bool]):
    """Simple boolean codec."""

    def encode(self, item: bool) -> bytes:
        return struct.pack("<?", item)

    def decode(self, data: Buffer) -> bool:
        return struct.unpack_from("<?", data, 0)[0]


class DataclassCodec[T](Codec[T]):
    """Codec for dataclass instances with typed fields.
    
    Provides type-safe attribute access while using the same efficient wire format
    as DictCodec. This is a thin wrapper that converts between dataclass instances
    and dicts.
    
    Example:
        >>> from dataclasses import dataclass
        >>> from tinman.codecs import DataclassCodec, NumpyArrayCodec, StringCodec
        >>> import numpy as np
        >>> 
        >>> @dataclass
        >>> class SensorReading:
        ...     data: np.ndarray
        ...     label: str
        ...     confidence: int
        >>> 
        >>> # Define schema (same as DictCodec)
        >>> schema = {
        ...     'data': NumpyArrayCodec(),
        ...     'label': StringCodec(),
        ...     'confidence': IntCodec(),
        ... }
        >>> codec = DataclassCodec(SensorReading, schema)
        >>> 
        >>> # Write dataclass instances
        >>> reading = SensorReading(
        ...     data=np.array([1, 2, 3]),
        ...     label='sensor_a',
        ...     confidence=95
        ... )
        >>> write(reading)
        >>> 
        >>> # Read back as dataclass instances (type-safe!)
        >>> async for _, reading in oblog.read_channel("sensors"):
        ...     print(reading.data)        # Attribute access
        ...     print(reading.label)       # IDE autocomplete works!
        ...     print(reading.confidence)  # Type checker validates
    
    Note:
        - Wire format is identical to DictCodec
        - Files written with DictCodec can be read with DataclassCodec and vice versa
        - Zero-copy semantics preserved for arrays in dataclass fields
        - Supports allow_missing for optional fields
    """

    def __init__(
        self,
        dataclass_type: type[T],
        schema: Mapping[str, Codec],
        *,
        allow_missing: bool = False,
    ):
        """Initialize with a dataclass type and field codecs.
        
        Args:
            dataclass_type: The dataclass type to encode/decode.
            schema: Dict mapping field names to their codecs.
                    Must match the dataclass field names.
            allow_missing: If True, missing fields are allowed (for optional fields).
                          Default is False (all fields required).
        
        Raises:
            TypeError: If dataclass_type is not a dataclass.
        """
        if not is_dataclass(dataclass_type):
            raise TypeError(
                f"{dataclass_type} is not a dataclass. "
                f"Use @dataclass decorator or pass a dataclass type."
            )
        self.dataclass_type = dataclass_type
        self.dict_codec = DictCodec(schema, allow_missing=allow_missing)

    def encode(self, item: T) -> bytes:
        """Encode dataclass instance to bytes.
        
        Args:
            item: Dataclass instance to encode.
            
        Returns:
            Bytes containing encoded dataclass data.
        """
        from dataclasses import asdict
        
        # Convert dataclass to dict, then use DictCodec
        d = asdict(item)  # type: ignore[arg-type]
        return self.dict_codec.encode(d)

    def decode(self, data: Buffer) -> T:
        """Decode buffer to dataclass instance (zero-copy where possible).
        
        Args:
            data: Buffer/memoryview containing encoded dataclass.
            
        Returns:
            Dataclass instance with decoded fields. If field codecs support
            zero-copy (like NumpyArrayCodec), those fields will be views.
        """
        # Use DictCodec to decode to dict, then construct dataclass
        d = self.dict_codec.decode(data)
        return self.dataclass_type(**d)
