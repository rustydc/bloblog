"""Benchmarks for codec encode/decode performance."""

import pickle
from collections.abc import Buffer

import pytest

from bloblog.codecs import Codec, PickleCodec


class SimpleStringCodec(Codec[str]):
    """Simple UTF-8 string codec for comparison."""

    def encode(self, item: str) -> bytes:
        return item.encode("utf-8")

    def decode(self, data: Buffer) -> str:
        return bytes(data).decode("utf-8")


class BinaryCodec(Codec[bytes]):
    """No-op codec for raw bytes."""

    def encode(self, item: bytes) -> bytes:
        return item

    def decode(self, data: Buffer) -> bytes:
        return bytes(data)


class TestCodecPerformance:
    """Benchmark different codec implementations."""

    @pytest.mark.benchmark(group="codec-string")
    def test_string_codec_encode(self, benchmark):
        """Benchmark string encoding."""
        codec = SimpleStringCodec()
        data = "The quick brown fox jumps over the lazy dog" * 10

        result = benchmark(codec.encode, data)
        assert isinstance(result, bytes)

    @pytest.mark.benchmark(group="codec-string")
    def test_string_codec_decode(self, benchmark):
        """Benchmark string decoding."""
        codec = SimpleStringCodec()
        data = b"The quick brown fox jumps over the lazy dog" * 10

        result = benchmark(codec.decode, data)
        assert isinstance(result, str)

    @pytest.mark.benchmark(group="codec-pickle")
    def test_pickle_codec_simple_encode(self, benchmark):
        """Benchmark pickle encoding for simple types."""
        codec = PickleCodec()
        data = {"key": "value", "count": 42, "items": [1, 2, 3, 4, 5]}

        result = benchmark(codec.encode, data)
        assert isinstance(result, bytes)

    @pytest.mark.benchmark(group="codec-pickle")
    def test_pickle_codec_simple_decode(self, benchmark):
        """Benchmark pickle decoding for simple types."""
        codec = PickleCodec()
        data = pickle.dumps({"key": "value", "count": 42, "items": [1, 2, 3, 4, 5]})

        result = benchmark(codec.decode, data)
        assert isinstance(result, dict)

    @pytest.mark.benchmark(group="codec-pickle-complex")
    def test_pickle_codec_complex_encode(self, benchmark):
        """Benchmark pickle with more complex structures."""
        codec = PickleCodec()
        # Simulate a sensor reading
        data = {
            "timestamp": 1234567890.123,
            "position": {"x": 1.5, "y": 2.3, "z": 0.8},
            "velocity": {"x": 0.1, "y": -0.2, "z": 0.0},
            "sensors": {
                "lidar": [1.2, 3.4, 5.6, 7.8, 9.0] * 20,  # 100 readings
                "camera": b"fake_image_data" * 100,
            },
        }

        result = benchmark(codec.encode, data)
        assert isinstance(result, bytes)

    @pytest.mark.benchmark(group="codec-pickle-complex")
    def test_pickle_codec_complex_decode(self, benchmark):
        """Benchmark pickle decoding complex structures."""
        codec = PickleCodec()
        data_dict = {
            "timestamp": 1234567890.123,
            "position": {"x": 1.5, "y": 2.3, "z": 0.8},
            "velocity": {"x": 0.1, "y": -0.2, "z": 0.0},
            "sensors": {
                "lidar": [1.2, 3.4, 5.6, 7.8, 9.0] * 20,
                "camera": b"fake_image_data" * 100,
            },
        }
        data = pickle.dumps(data_dict)

        result = benchmark(codec.decode, data)
        assert isinstance(result, dict)

    @pytest.mark.benchmark(group="codec-binary")
    def test_binary_codec_noop(self, benchmark):
        """Benchmark no-op binary codec (baseline)."""
        codec = BinaryCodec()
        data = b"x" * 10_000

        result = benchmark(codec.encode, data)
        assert result == data

    @pytest.mark.benchmark(group="codec-overhead")
    def test_codec_roundtrip_string(self, benchmark):
        """Benchmark full encode-decode cycle for strings."""
        codec = SimpleStringCodec()
        data = "Test message" * 100

        def roundtrip():
            encoded = codec.encode(data)
            decoded = codec.decode(encoded)
            return decoded

        result = benchmark(roundtrip)
        assert result == data

    @pytest.mark.benchmark(group="codec-overhead")
    def test_codec_roundtrip_pickle(self, benchmark):
        """Benchmark full encode-decode cycle for pickle."""
        codec = PickleCodec()
        data = {"message": "test", "count": 123, "values": [1, 2, 3, 4, 5]}

        def roundtrip():
            encoded = codec.encode(data)
            decoded = codec.decode(encoded)
            return decoded

        result = benchmark(roundtrip)
        assert result == data

    @pytest.mark.benchmark(group="memoryview-conversion")
    def test_memoryview_to_bytes_conversion(self, benchmark):
        """Benchmark cost of converting memoryview to bytes in decode."""
        data = b"x" * 10_000
        mv = memoryview(data)

        result = benchmark(bytes, mv)
        assert result == data

    @pytest.mark.benchmark(group="memoryview-conversion")
    def test_decode_from_memoryview_vs_bytes(self, benchmark):
        """Compare decoding from memoryview vs bytes."""
        codec = SimpleStringCodec()
        data = b"Test message" * 100

        # Test with memoryview (current API)
        def decode_memoryview():
            mv = memoryview(data)
            return codec.decode(mv)

        result = benchmark(decode_memoryview)
        assert isinstance(result, str)
