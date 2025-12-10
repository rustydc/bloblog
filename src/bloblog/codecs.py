"""Codec system for encoding/decoding channel data in bloblog."""

from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod
from collections.abc import Buffer

# Global codec registry - populated automatically when Codec subclasses are defined
_CODEC_REGISTRY: dict[str, type[Codec]] = {}


class Codec[T](ABC):
    """Abstract base class for encoding/decoding channel data.

    Codecs are automatically registered in the global registry when defined.
    This allows log files to be self-describing with codec metadata.

    To suppress auto-registration (e.g., for PickleCodec), set _auto_register=False
    in the class definition.
    """

    def __init_subclass__(cls, _auto_register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register when subclass is defined (unless suppressed)
        if _auto_register:
            qualified_name = f"{cls.__module__}.{cls.__qualname__}"
            _CODEC_REGISTRY[qualified_name] = cls

    @abstractmethod
    def encode(self, item: T) -> Buffer: ...

    @abstractmethod
    def decode(self, data: Buffer) -> T: ...

    @classmethod
    def get_qualified_name(cls) -> str:
        """Get the fully qualified name of this codec class."""
        return f"{cls.__module__}.{cls.__qualname__}"


class PickleCodec[T](Codec[T], _auto_register=False):
    """Generic codec that uses pickle for serialization.

    This codec is NOT registered by default for security reasons.
    Call enable_pickle_codec() to register it if needed.

    Works for any picklable Python object, but less efficient than specialized codecs
    and carries pickle's security risks.
    """

    def encode(self, item: T) -> bytes:
        return pickle.dumps(item)

    def decode(self, data: Buffer) -> T:
        return pickle.loads(bytes(data))


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows registered codec classes and safe builtins."""

    def find_class(self, module, name):
        # This is called for every class pickle tries to instantiate
        qualified_name = f"{module}.{name}"

        # Allow registered codecs
        if qualified_name in _CODEC_REGISTRY:
            return _CODEC_REGISTRY[qualified_name]

        # Allow safe builtins that codecs might use
        safe_modules = {
            "builtins": {
                "int",
                "float",
                "str",
                "bytes",
                "list",
                "dict",
                "tuple",
                "set",
                "frozenset",
                "bool",
                "NoneType",
            },
            "collections": {"OrderedDict", "defaultdict", "Counter"},
        }
        if module in safe_modules and name in safe_modules[module]:
            return super().find_class(module, name)

        raise pickle.UnpicklingError(
            f"Attempted to unpickle unauthorized class: {qualified_name}. "
            f"Only registered Codec subclasses are allowed."
        )


def safe_unpickle_codec(data: bytes) -> Codec:
    """Safely unpickle a codec, only allowing registered classes.

    Args:
        data: Pickled codec bytes.

    Returns:
        The unpickled Codec instance.

    Raises:
        ValueError: If codec class is not registered.
        pickle.UnpicklingError: If unpickling attempts to load unauthorized class.
    """
    unpickler = _RestrictedUnpickler(io.BytesIO(data))
    codec = unpickler.load()

    # Verify it's a registered codec
    actual_classname = f"{codec.__class__.__module__}.{codec.__class__.__qualname__}"
    if actual_classname not in _CODEC_REGISTRY:
        raise ValueError(
            f"Codec class not registered: {actual_classname}. "
            f"Make sure the codec module is imported."
        )

    if not isinstance(codec, Codec):
        raise ValueError(f"Unpickled object is not a Codec: {type(codec)}")

    return codec


def enable_pickle_codec() -> None:
    """Register PickleCodec in the global codec registry.

    Call this function to enable pickle-based serialization for channels.
    Note: This should only be used when you trust the data source, as pickle
    has known security vulnerabilities.
    """
    qualified_name = PickleCodec.get_qualified_name()
    _CODEC_REGISTRY[qualified_name] = PickleCodec
