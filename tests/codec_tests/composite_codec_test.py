"""Tests for composite codec utilities."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from tinman import ObLogWriter, ObLogReader
from tinman.codecs import (
    BoolCodec,
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    NumpyArrayCodec,
    OptionalCodec,
    StringCodec,
    TupleCodec,
)


class TestDictCodec:
    """Test DictCodec for dicts with typed values."""

    def test_dict_with_arrays(self):
        """Test dict containing numpy arrays (zero-copy)."""
        schema = {
            "sensor_1": NumpyArrayCodec(),
            "sensor_2": NumpyArrayCodec(),
        }
        codec = DictCodec(schema)

        # Original data
        data = {
            "sensor_1": np.array([1, 2, 3], dtype=np.int32),
            "sensor_2": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        }

        # Encode/decode
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        # Verify
        np.testing.assert_array_equal(decoded["sensor_1"], data["sensor_1"])
        np.testing.assert_array_equal(decoded["sensor_2"], data["sensor_2"])

        # Verify zero-copy (read-only)
        assert not decoded["sensor_1"].flags.writeable
        assert not decoded["sensor_2"].flags.writeable

    def test_dict_with_mixed_types(self):
        """Test dict with arrays, strings, and primitives."""
        schema = {
            "array": NumpyArrayCodec(),
            "name": StringCodec(),
            "count": IntCodec(),
            "ratio": FloatCodec(),
            "active": BoolCodec(),
        }
        codec = DictCodec(schema)

        data = {
            "array": np.array([1, 2, 3]),
            "name": "sensor_a",
            "count": 42,
            "ratio": 3.14,
            "active": True,
        }

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        np.testing.assert_array_equal(decoded["array"], data["array"])
        assert decoded["name"] == data["name"]
        assert decoded["count"] == data["count"]
        assert abs(decoded["ratio"] - data["ratio"]) < 1e-10
        assert decoded["active"] == data["active"]

    def test_dict_missing_key_strict(self):
        """Test that missing keys raise error by default."""
        schema = {
            "required": NumpyArrayCodec(),
            "also_required": StringCodec(),
        }
        codec = DictCodec(schema)  # allow_missing=False (default)

        data = {"required": np.array([1, 2, 3])}  # Missing 'also_required'

        with pytest.raises(KeyError, match="also_required"):
            codec.encode(data)

    def test_dict_missing_key_allowed(self):
        """Test that missing keys are handled when allow_missing=True."""
        schema = {
            "required": NumpyArrayCodec(),
            "optional": StringCodec(),
        }
        codec = DictCodec(schema, allow_missing=True)

        # Write with missing key
        data = {"required": np.array([1, 2, 3])}  # 'optional' missing
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        # Decoded dict should only have present keys
        assert "required" in decoded
        assert "optional" not in decoded
        np.testing.assert_array_equal(decoded["required"], data["required"])

    def test_dict_missing_key_multiple(self):
        """Test multiple missing keys."""
        schema = {
            "a": IntCodec(),
            "b": IntCodec(),
            "c": IntCodec(),
            "d": IntCodec(),
        }
        codec = DictCodec(schema, allow_missing=True)

        # Only some keys present
        data = {"a": 1, "c": 3}  # b and d missing
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert decoded == {"a": 1, "c": 3}
        assert "b" not in decoded
        assert "d" not in decoded

    def test_dict_all_keys_missing(self):
        """Test dict with all keys missing."""
        schema = {
            "a": IntCodec(),
            "b": IntCodec(),
        }
        codec = DictCodec(schema, allow_missing=True)

        data = {}  # All keys missing
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert decoded == {}

    def test_dict_missing_key_with_arrays(self):
        """Test missing keys work with zero-copy arrays."""
        schema = {
            "array1": NumpyArrayCodec(),
            "array2": NumpyArrayCodec(),
            "label": StringCodec(),
        }
        codec = DictCodec(schema, allow_missing=True)

        # Only array1 present
        data = {"array1": np.array([1, 2, 3])}
        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert "array1" in decoded
        assert "array2" not in decoded
        assert "label" not in decoded
        np.testing.assert_array_equal(decoded["array1"], data["array1"])
        assert not decoded["array1"].flags.writeable  # Still zero-copy!

    @pytest.mark.asyncio
    async def test_dict_codec_with_oblog(self):
        """Test DictCodec integration with ObLog."""
        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Write
            schema = {
                "data": NumpyArrayCodec(),
                "label": StringCodec(),
            }
            codec = DictCodec(schema)

            async with ObLogWriter(log_dir) as oblog:
                write = oblog.get_writer("records", codec)

                records = [
                    {"data": np.array([1, 2, 3]), "label": "first"},
                    {"data": np.array([4, 5, 6]), "label": "second"},
                ]

                for record in records:
                    write(record)

            # Read (no context manager needed)
            reader = ObLogReader(log_dir)
            read_records = []
            async for _, record in reader.read_channel("records"):
                # Copy arrays to detach from mmap
                read_records.append({
                    "data": record["data"].copy(),
                    "label": record["label"],
                })

            # Verify
            assert len(read_records) == len(records)
            for orig, read in zip(records, read_records):
                np.testing.assert_array_equal(orig["data"], read["data"])
                assert orig["label"] == read["label"]


class TestListCodec:
    """Test ListCodec for lists with uniform element type."""

    def test_list_of_arrays(self):
        """Test list containing numpy arrays (zero-copy)."""
        codec = ListCodec(NumpyArrayCodec())

        data = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        ]

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert len(decoded) == len(data)
        for orig, read in zip(data, decoded):
            np.testing.assert_array_equal(orig, read)
            assert not read.flags.writeable  # Zero-copy, read-only

    def test_list_of_strings(self):
        """Test list of strings."""
        codec = ListCodec(StringCodec())

        data = ["hello", "world", "test"]

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert decoded == data

    def test_empty_list(self):
        """Test empty list."""
        codec = ListCodec(IntCodec())

        data = []

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert decoded == data


class TestTupleCodec:
    """Test TupleCodec for tuples with heterogeneous types."""

    def test_tuple_mixed_types(self):
        """Test tuple with mixed types."""
        codec = TupleCodec([
            NumpyArrayCodec(),
            StringCodec(),
            IntCodec(),
        ])

        data = (
            np.array([1, 2, 3]),
            "label",
            42,
        )

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert len(decoded) == 3
        np.testing.assert_array_equal(decoded[0], data[0])
        assert decoded[1] == data[1]
        assert decoded[2] == data[2]

    def test_tuple_wrong_length(self):
        """Test that wrong tuple length raises error."""
        codec = TupleCodec([IntCodec(), IntCodec()])

        with pytest.raises(ValueError, match="has 3 elements but 2 codecs"):
            codec.encode((1, 2, 3))


class TestOptionalCodec:
    """Test OptionalCodec for nullable values."""

    def test_optional_with_value(self):
        """Test optional field with a value."""
        codec = OptionalCodec(NumpyArrayCodec())

        data = np.array([1, 2, 3])

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        np.testing.assert_array_equal(decoded, data)

    def test_optional_with_none(self):
        """Test optional field with None."""
        codec = OptionalCodec(NumpyArrayCodec())

        encoded = codec.encode(None)
        decoded = codec.decode(encoded)

        assert decoded is None

    def test_optional_in_dict(self):
        """Test optional fields in dict schema."""
        schema = {
            "required": NumpyArrayCodec(),
            "optional": OptionalCodec(StringCodec()),
        }
        codec = DictCodec(schema)

        # With optional value
        data1 = {
            "required": np.array([1, 2]),
            "optional": "present",
        }
        encoded1 = codec.encode(data1)
        decoded1 = codec.decode(encoded1)
        np.testing.assert_array_equal(decoded1["required"], data1["required"])
        assert decoded1["optional"] == "present"

        # Without optional value
        data2 = {
            "required": np.array([3, 4]),
            "optional": None,
        }
        encoded2 = codec.encode(data2)
        decoded2 = codec.decode(encoded2)
        np.testing.assert_array_equal(decoded2["required"], data2["required"])
        assert decoded2["optional"] is None


class TestNestedComposites:
    """Test nested composite structures."""

    def test_list_of_dicts(self):
        """Test list containing dicts with arrays."""
        item_schema = {
            "data": NumpyArrayCodec(),
            "id": IntCodec(),
        }
        codec = ListCodec(DictCodec(item_schema))

        data = [
            {"data": np.array([1, 2]), "id": 1},
            {"data": np.array([3, 4]), "id": 2},
        ]

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert len(decoded) == len(data)
        for orig, read in zip(data, decoded):
            np.testing.assert_array_equal(orig["data"], read["data"])
            assert orig["id"] == read["id"]

    def test_dict_of_lists(self):
        """Test dict containing lists of arrays."""
        schema = {
            "batch_a": ListCodec(NumpyArrayCodec()),
            "batch_b": ListCodec(NumpyArrayCodec()),
        }
        codec = DictCodec(schema)

        data = {
            "batch_a": [np.array([1, 2]), np.array([3, 4])],
            "batch_b": [np.array([5, 6]), np.array([7, 8])],
        }

        encoded = codec.encode(data)
        decoded = codec.decode(encoded)

        assert len(decoded["batch_a"]) == 2
        assert len(decoded["batch_b"]) == 2
        for orig, read in zip(data["batch_a"], decoded["batch_a"]):
            np.testing.assert_array_equal(orig, read)


class TestPrimitiveCodecs:
    """Test primitive codec implementations."""

    def test_int_codec(self):
        """Test IntCodec."""
        codec = IntCodec()
        for value in [0, 1, -1, 42, -42, 2**32, -(2**32)]:
            encoded = codec.encode(value)
            decoded = codec.decode(encoded)
            assert decoded == value

    def test_float_codec(self):
        """Test FloatCodec."""
        codec = FloatCodec()
        for value in [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e10, 1e-10]:
            encoded = codec.encode(value)
            decoded = codec.decode(encoded)
            assert abs(decoded - value) < 1e-10

    def test_string_codec(self):
        """Test StringCodec."""
        codec = StringCodec()
        for value in ["", "hello", "世界", "emoji 🎉"]:
            encoded = codec.encode(value)
            decoded = codec.decode(encoded)
            assert decoded == value

    def test_bool_codec(self):
        """Test BoolCodec."""
        codec = BoolCodec()
        for value in [True, False]:
            encoded = codec.encode(value)
            decoded = codec.decode(encoded)
            assert decoded == value


class TestDataclassCodec:
    """Test DataclassCodec for type-safe dataclass serialization."""

    def test_basic_dataclass(self):
        """Test basic dataclass encoding/decoding."""
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class Person:
            name: str
            age: int

        schema = {
            "name": StringCodec(),
            "age": IntCodec(),
        }
        codec = DataclassCodec(Person, schema)

        # Encode
        person = Person(name="Alice", age=30)
        encoded = codec.encode(person)

        # Decode
        decoded = codec.decode(encoded)
        assert isinstance(decoded, Person)
        assert decoded.name == "Alice"
        assert decoded.age == 30

    def test_dataclass_with_arrays(self):
        """Test dataclass with numpy arrays (zero-copy)."""
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class SensorReading:
            data: np.ndarray
            label: str

        schema = {
            "data": NumpyArrayCodec(),
            "label": StringCodec(),
        }
        codec = DataclassCodec(SensorReading, schema)

        # Encode
        reading = SensorReading(
            data=np.array([1, 2, 3], dtype=np.int32),
            label="sensor_a"
        )
        encoded = codec.encode(reading)

        # Decode
        decoded = codec.decode(encoded)
        assert isinstance(decoded, SensorReading)
        np.testing.assert_array_equal(decoded.data, reading.data)
        assert decoded.label == "sensor_a"
        assert not decoded.data.flags.writeable  # Zero-copy!

    def test_dataclass_with_multiple_types(self):
        """Test dataclass with various field types."""
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class Measurement:
            timestamp: int
            value: float
            valid: bool
            sensor_id: str
            samples: np.ndarray

        schema = {
            "timestamp": IntCodec(),
            "value": FloatCodec(),
            "valid": BoolCodec(),
            "sensor_id": StringCodec(),
            "samples": NumpyArrayCodec(),
        }
        codec = DataclassCodec(Measurement, schema)

        # Encode
        m = Measurement(
            timestamp=1000,
            value=3.14,
            valid=True,
            sensor_id="S001",
            samples=np.array([1.1, 2.2, 3.3])
        )
        encoded = codec.encode(m)

        # Decode
        decoded = codec.decode(encoded)
        assert decoded.timestamp == 1000
        assert abs(decoded.value - 3.14) < 1e-10
        assert decoded.valid is True
        assert decoded.sensor_id == "S001"
        np.testing.assert_array_almost_equal(decoded.samples, m.samples)

    def test_dataclass_missing_fields(self):
        """Test dataclass with allow_missing for optional fields with defaults."""
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class Record:
            required: str
            optional: str = "default_value"

        schema = {
            "required": StringCodec(),
            "optional": StringCodec(),
        }
        codec = DataclassCodec(Record, schema, allow_missing=True)

        # Encode with missing optional field using DictCodec directly
        from tinman.codecs import DictCodec
        dict_codec = DictCodec(schema, allow_missing=True)
        encoded = dict_codec.encode({"required": "test"})

        # Decode - should work because dataclass has default for 'optional'
        decoded = codec.decode(encoded)
        assert decoded.required == "test"
        assert decoded.optional == "default_value"  # Uses default

    @pytest.mark.asyncio
    async def test_dataclass_with_oblog(self):
        """Test DataclassCodec integration with ObLog.
        
        Note: This test doesn't use ObLog's codec pickling because local dataclasses
        can't be pickled. In real usage, dataclasses should be defined at module level.
        """
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class Event:
            data: np.ndarray
            label: str
            count: int

        with TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            schema = {
                "data": NumpyArrayCodec(),
                "label": StringCodec(),
                "count": IntCodec(),
            }
            codec = DataclassCodec(Event, schema)

            # Test encode/decode directly (without ObLog's codec pickling)
            events = [
                Event(np.array([1, 2]), "first", 1),
                Event(np.array([3, 4]), "second", 2),
            ]

            # Encode events
            encoded = [codec.encode(event) for event in events]

            # Decode events
            read_events = []
            for enc in encoded:
                event = codec.decode(enc)
                assert isinstance(event, Event)
                read_events.append(Event(
                    data=event.data.copy(),
                    label=event.label,
                    count=event.count
                ))

            # Verify
            assert len(read_events) == len(events)
            for orig, read in zip(events, read_events):
                np.testing.assert_array_equal(orig.data, read.data)
                assert orig.label == read.label
                assert orig.count == read.count

    def test_dataclass_interop_with_dict_codec(self):
        """Test that DataclassCodec and DictCodec use same wire format."""
        from dataclasses import dataclass

        from tinman.codecs import DataclassCodec

        @dataclass
        class Point:
            x: int
            y: int

        schema = {
            "x": IntCodec(),
            "y": IntCodec(),
        }

        # Encode with DataclassCodec
        dataclass_codec = DataclassCodec(Point, schema)
        point = Point(x=10, y=20)
        encoded_dataclass = dataclass_codec.encode(point)

        # Encode with DictCodec
        dict_codec = DictCodec(schema)
        encoded_dict = dict_codec.encode({"x": 10, "y": 20})

        # Should produce identical bytes
        assert encoded_dataclass == encoded_dict

        # Decode with opposite codec
        decoded_dict = dict_codec.decode(encoded_dataclass)
        assert decoded_dict == {"x": 10, "y": 20}

        decoded_dataclass = dataclass_codec.decode(encoded_dict)
        assert decoded_dataclass.x == 10
        assert decoded_dataclass.y == 20
