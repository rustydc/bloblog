"""Protocol Buffer codec for tinman.

Provides efficient serialization for protobuf messages using the standard
protobuf wire format.
"""

from __future__ import annotations

from collections.abc import Buffer
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from google.protobuf.message import Message

from ..oblog import Codec

# Type variable bound to protobuf Message for proper type checking
T = TypeVar("T", bound="Message")


class ProtobufCodec(Codec[T]):
    """Codec for Protocol Buffer messages.
    
    Uses standard protobuf serialization (SerializeToString/ParseFromString).
    Efficient and compact binary format with built-in schema evolution support.
    
    Note: This is not zero-copy like NumpyArrayCodec - protobuf deserialization
    creates new message objects. For zero-copy arrays, use NumpyArrayCodec directly
    or combine with DictCodec.
    
    Example:
        >>> from tinman.codecs import ProtobufCodec
        >>> from tinman import ObLog
        >>> from pathlib import Path
        >>> from your_proto_pb2 import SensorReading  # Your .proto generated class
        >>> 
        >>> # Write protobuf messages
        >>> oblog = ObLog(Path("logs"))
        >>> codec = ProtobufCodec(SensorReading)
        >>> write = oblog.get_writer("sensors", codec)
        >>> 
        >>> msg = SensorReading(temperature=20.5, humidity=65, sensor_id="s1")
        >>> write(msg)
        >>> await oblog.close()
        >>> 
        >>> # Read protobuf messages
        >>> oblog = ObLog(Path("logs"))
        >>> async for timestamp, msg in oblog.read_channel("sensors"):
        ...     print(f"Temp: {msg.temperature}, Humidity: {msg.humidity}")
        >>> await oblog.close()
    
    Combining with zero-copy arrays:
        >>> from tinman.codecs import DictCodec, NumpyArrayCodec, ProtobufCodec
        >>> import numpy as np
        >>> 
        >>> # Use DictCodec to combine protobuf metadata with zero-copy arrays
        >>> schema = {
        ...     'metadata': ProtobufCodec(SensorMetadata),  # Protobuf for metadata
        ...     'readings': NumpyArrayCodec(),               # Zero-copy for arrays
        ... }
        >>> codec = DictCodec(schema)
        >>> 
        >>> # Write mixed data
        >>> data = {
        ...     'metadata': SensorMetadata(sensor_id="s1", location="lab"),
        ...     'readings': np.array([20.1, 20.3, 20.5, 20.7])
        ... }
        >>> write(data)
    """

    def __init__(self, message_class: type[T]):
        """Initialize with protobuf message class.
        
        Args:
            message_class: The generated protobuf message class.
                          This should be a class generated from your .proto file,
                          e.g., `from your_proto_pb2 import YourMessage`
        
        Example:
            >>> from your_proto_pb2 import SensorReading
            >>> codec = ProtobufCodec(SensorReading)
        """
        self.message_class = message_class

    def encode(self, item: T) -> bytes:
        """Encode protobuf message to bytes.
        
        Args:
            item: Protobuf message instance to encode.
            
        Returns:
            Bytes containing the protobuf wire format.
            
        Raises:
            AttributeError: If item doesn't have SerializeToString method.
        """
        return item.SerializeToString()

    def decode(self, data: Buffer) -> T:
        """Decode bytes to protobuf message.
        
        Args:
            data: Buffer/memoryview containing protobuf wire format bytes.
            
        Returns:
            Decoded protobuf message instance.
            
        Note:
            This creates a new message object (not zero-copy).
            The message is fully parsed and owns its data.
        """
        msg = self.message_class()
        # Convert Buffer to bytes if needed for ParseFromString
        msg.ParseFromString(bytes(data))
        return msg
