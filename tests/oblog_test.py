"""Tests for oblog module (codecs and encoded readers/writers)."""

from typing import Annotated

import pytest

from tinman import In, Out
from tinman.runtime import run_nodes
from tinman.oblog import PickleCodec


class TestDefaultCodec:
    """Test that outputs can use default PickleCodec."""

    @pytest.mark.asyncio
    async def test_default_codec_with_strings(self):
        """Test default codec works with strings."""
        results = []

        async def producer(output: Annotated[Out[str], "test"]):
            await output.publish("hello")
            await output.publish("world")

        async def consumer(input: Annotated[In[str], "test"]):
            async for msg in input:
                results.append(msg)

        await run_nodes([producer, consumer])
        assert results == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_default_codec_with_dicts(self):
        """Test default codec works with dictionaries."""
        results = []

        async def producer(output: Annotated[Out[dict], "test"]):
            await output.publish({"a": 1, "b": 2})
            await output.publish({"c": 3, "d": 4})

        async def consumer(input: Annotated[In[dict], "test"]):
            async for msg in input:
                results.append(msg)

        await run_nodes([producer, consumer])
        assert results == [{"a": 1, "b": 2}, {"c": 3, "d": 4}]

    @pytest.mark.asyncio
    async def test_default_codec_with_complex_types(self):
        """Test default codec works with complex nested types."""
        results = []

        async def producer(output: Annotated[Out[dict], "test"]):
            data = {
                "nested": {"list": [1, 2, 3], "tuple": (4, 5)},
                "set": {6, 7, 8},
                "mixed": ["a", 1, True, None],
            }
            await output.publish(data)

        async def consumer(input: Annotated[In[dict], "test"]):
            async for msg in input:
                results.append(msg)

        await run_nodes([producer, consumer])
        assert len(results) == 1
        assert results[0]["nested"]["list"] == [1, 2, 3]
        assert results[0]["nested"]["tuple"] == (4, 5)
        assert results[0]["set"] == {6, 7, 8}
        assert results[0]["mixed"] == ["a", 1, True, None]

    @pytest.mark.asyncio
    async def test_pickle_codec_class_exists(self):
        """Test that PickleCodec can be instantiated directly."""
        codec = PickleCodec()

        # Test encode/decode
        data = {"test": "value", "number": 42}
        encoded = codec.encode(data)
        assert isinstance(encoded, bytes)

        decoded = codec.decode(encoded)
        assert decoded == data

    @pytest.mark.asyncio
    async def test_mixed_explicit_and_default_codecs(self):
        """Test that explicit and default codecs can coexist."""
        from collections.abc import Buffer

        from tinman import Codec

        class CustomCodec(Codec[str]):
            def encode(self, item: str) -> bytes:
                return f"CUSTOM:{item}".encode()

            def decode(self, data: Buffer) -> str:
                return bytes(data).decode().replace("CUSTOM:", "")

        results_default = []
        results_custom = []

        async def producer1(output: Annotated[Out[str], "default"]):
            await output.publish("default")

        async def producer2(output: Annotated[Out[str], "custom", CustomCodec()]):
            await output.publish("custom")

        async def consumer1(input: Annotated[In[str], "default"]):
            async for msg in input:
                results_default.append(msg)

        async def consumer2(input: Annotated[In[str], "custom"]):
            async for msg in input:
                results_custom.append(msg)

        await run_nodes([producer1, producer2, consumer1, consumer2])

        assert results_default == ["default"]
        assert results_custom == ["custom"]
