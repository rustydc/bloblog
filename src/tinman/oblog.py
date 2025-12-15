"""Object logging with automatic codec serialization."""

from __future__ import annotations

import pickle
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

from .bloblog import BlobLog


class ObLog:
    """Object log directory with automatic encoding/decoding.

    Wraps BlobLog to handle codec metadata and object serialization.
    Each channel stores its codec as the first record.
    """

    def __init__(self, log_dir: Path):
        self.blob_log = BlobLog(log_dir)
        self._codecs: dict[str, Codec] = {}

    def get_writer(self, channel: str, codec: Codec) -> Callable[[object], None]:
        """Get a write function for a channel with automatic encoding."""
        if channel in self._codecs:
            raise ValueError(f"Channel '{channel}' already has a writer")

        self._codecs[channel] = codec
        raw_writer = self.blob_log.get_writer(channel)

        # Write codec metadata as first record
        codec_bytes = pickle.dumps(codec)
        raw_writer(codec_bytes)

        def write(item: object) -> None:
            data = codec.encode(item)
            raw_writer(data)

        return write

    async def read_channel(self, channel: str) -> AsyncGenerator[tuple[int, object], None]:
        """Read a channel with automatic decoding.

        The codec is auto-detected from the first record.
        """
        # Read first record to get codec
        first = True
        codec = None

        async for timestamp, data in self.blob_log.read_channel(channel):
            if first:
                codec = safe_unpickle_codec(bytes(data))
                first = False
                continue

            assert codec is not None
            item = codec.decode(data)
            yield timestamp, item

    async def read_codec(self, channel: str) -> Codec:
        """Read just the codec for a channel without reading data records.

        This reads only the first record (which contains the pickled codec)
        and returns it. Useful for inspecting channel metadata.

        Args:
            channel: Channel name to read codec from.

        Returns:
            The Codec instance stored in the channel.

        Raises:
            FileNotFoundError: If channel log file doesn't exist.
            ValueError: If log file is empty or malformed.

        Example:
            >>> oblog = ObLog(Path("logs"))
            >>> codec = await oblog.read_codec("camera")
            >>> print(f"Camera uses codec: {type(codec).__name__}")
        """
        async for _timestamp, data in self.blob_log.read_channel(channel):
            # First record is always the codec
            return safe_unpickle_codec(bytes(data))
        raise ValueError(f"Channel '{channel}' log file is empty")

    async def close(self) -> None:
        """Close the underlying BlobLog."""
        await self.blob_log.close()


class Codec[T]:
    """Base class for encoding/decoding objects to/from bytes.

    Subclasses should implement encode() and decode() methods.
    Codec instances are pickled and stored as the first record in each log file.
    """

    def __init_subclass__(cls, _auto_register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register when subclass is defined (unless suppressed)
        if _auto_register:
            qualified_name = f"{cls.__module__}.{cls.__qualname__}"
            _CODEC_REGISTRY[qualified_name] = cls

    def encode(self, item: T) -> bytes:
        """Encode an object to bytes."""
        raise NotImplementedError

    def decode(self, data: bytes) -> T:
        """Decode bytes to an object."""
        raise NotImplementedError

    @classmethod
    def get_qualified_name(cls) -> str:
        """Get the fully qualified name for codec registry."""
        return f"{cls.__module__}.{cls.__qualname__}"


# Global registry of codec classes by qualified name
_CODEC_REGISTRY: dict[str, type[Codec]] = {}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows registered Codec classes."""

    def find_class(self, module: str, name: str) -> type:
        qualified_name = f"{module}.{name}"
        if qualified_name not in _CODEC_REGISTRY:
            raise pickle.UnpicklingError(
                f"Codec '{qualified_name}' not registered. "
                f"Available: {list(_CODEC_REGISTRY.keys())}"
            )
        return _CODEC_REGISTRY[qualified_name]


def safe_unpickle_codec(data: bytes) -> Codec:
    """Safely unpickle a codec instance, only allowing registered types."""
    import io

    return _RestrictedUnpickler(io.BytesIO(data)).load()


class PickleCodec(Codec, _auto_register=False):
    """Codec that uses pickle for serialization.

    NOT auto-registered for security reasons. Call enable_pickle_codec() to use.
    """

    def encode(self, item: object) -> bytes:
        return pickle.dumps(item)

    def decode(self, data: bytes) -> object:
        return pickle.loads(data)


def enable_pickle_codec() -> None:
    """Register PickleCodec in the global codec registry.

    Call this function to enable pickle-based serialization for channels.
    Note: This should only be used when you trust the data source, as pickle
    has known security vulnerabilities.
    """
    qualified_name = PickleCodec.get_qualified_name()
    _CODEC_REGISTRY[qualified_name] = PickleCodec
