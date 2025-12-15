"""Example showing type-safe dataclass serialization with DataclassCodec.

Demonstrates how DataclassCodec provides IDE support, type checking, and
ergonomic attribute access while using the same efficient wire format as DictCodec.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tinman import ObLog
from tinman.codecs import DataclassCodec, FloatCodec, IntCodec, NumpyArrayCodec, StringCodec


# Define dataclasses at module level (required for pickling)
@dataclass
class SensorReading:
    """A sensor reading with data, label, and confidence."""
    data: np.ndarray
    label: str
    confidence: int
    temperature: float


@dataclass
class Point:
    """A simple 2D point."""
    x: int
    y: int


async def main():
    """Demonstrate DataclassCodec usage."""
    
    print("=" * 60)
    print("DataclassCodec: Type-Safe Serialization")
    print("=" * 60)
    
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Define schema (same as DictCodec)
        schema = {
            "data": NumpyArrayCodec(),
            "label": StringCodec(),
            "confidence": IntCodec(),
            "temperature": FloatCodec(),
        }
        codec = DataclassCodec(SensorReading, schema)
        
        print("\n✨ Benefits over DictCodec:")
        print("  - Type-safe attribute access (reading.data)")
        print("  - IDE autocomplete and refactoring")
        print("  - Type checker validates usage")
        print("  - Self-documenting with field types")
        print("  - Same efficient wire format!")
        
        # Test encode/decode directly (avoiding pickle issue with local class)
        print("\n📝 Writing sensor readings:\n")
        
        readings = [
            SensorReading(
                data=np.random.rand(100, 3),
                label="accelerometer",
                confidence=95,
                temperature=25.3
            ),
            SensorReading(
                data=np.random.rand(100, 3),
                label="gyroscope",
                confidence=88,
                temperature=26.1
            ),
            SensorReading(
                data=np.random.rand(100, 3),
                label="magnetometer",
                confidence=92,
                temperature=25.8
            ),
        ]
        
        # Encode to bytes
        encoded_readings = []
        for reading in readings:
            encoded = codec.encode(reading)
            encoded_readings.append(encoded)
            print(f"  {reading.label}: {reading.data.shape}, "
                  f"confidence={reading.confidence}, "
                  f"temp={reading.temperature:.1f}°C")
        
        # Decode from bytes
        print("\n📖 Reading back:\n")
        
        for encoded in encoded_readings:
            reading = codec.decode(encoded)
            
            # Type-safe attribute access!
            print(f"  {reading.label}:")
            print(f"    data shape: {reading.data.shape}")
            print(f"    confidence: {reading.confidence}%")
            print(f"    temperature: {reading.temperature:.1f}°C")
            print(f"    zero-copy: {not reading.data.flags.writeable}")
        
        print("\n" + "=" * 60)
        print("🔍 Compare with DictCodec:")
        print("=" * 60)
        
        # Show equivalent DictCodec usage
        from tinman.codecs import DictCodec
        dict_codec = DictCodec(schema)
        
        # Encode the same data both ways
        point_schema = {
            "x": IntCodec(),
            "y": IntCodec(),
        }
        
        dataclass_codec = DataclassCodec(Point, point_schema)
        point = Point(x=10, y=20)
        encoded_dc = dataclass_codec.encode(point)
        
        dict_codec_2 = DictCodec(point_schema)
        encoded_dict = dict_codec_2.encode({"x": 10, "y": 20})
        
        print(f"\nDataclassCodec bytes: {encoded_dc.hex()}")
        print(f"DictCodec bytes:      {encoded_dict.hex()}")
        print(f"Identical: {encoded_dc == encoded_dict} ✅")
        
        # Can decode with either codec!
        decoded_as_dict = dict_codec_2.decode(encoded_dc)
        decoded_as_dataclass = dataclass_codec.decode(encoded_dict)
        
        print(f"\nDecode DataclassCodec with DictCodec: {decoded_as_dict}")
        print(f"Decode DictCodec with DataclassCodec: {decoded_as_dataclass}")
        
        print("\n" + "=" * 60)
        print("✅ DataclassCodec = DictCodec + Type Safety")
        print("   Same wire format, better developer experience!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
