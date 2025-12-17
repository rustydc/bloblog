"""Image codec for webcam and other image data.

This codec efficiently serializes timestamped images using
NumpyArrayCodec for zero-copy image handling.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..oblog import Codec
from .numpy import NumpyArrayCodec


@dataclass
class Image:
    """A timestamped image.

    Attributes:
        timestamp_ns: Capture timestamp in nanoseconds
        seq: Sequential frame counter
        data: Image as numpy array (H, W, C) - typically BGR for OpenCV
    """

    timestamp_ns: int
    seq: int
    data: np.ndarray


class ImageCodec(Codec[Image]):
    """Codec for timestamped images.

    Efficiently serializes Image dataclass by:
    1. Storing timestamp and sequence number as fixed-size integers
    2. Using NumpyArrayCodec for the image data

    Format:
        - timestamp_ns: int64 (8 bytes)
        - seq: int64 (8 bytes)
        - image_data: NumpyArrayCodec encoded bytes

    Example:
        >>> from tinman.codecs.image import Image, ImageCodec
        >>> codec = ImageCodec()
        >>> img = Image(timestamp_ns=1000, seq=0, data=np.zeros((480, 640, 3), dtype=np.uint8))
        >>> data = codec.encode(img)
        >>> decoded = codec.decode(data)
    """

    def __init__(self):
        self._numpy_codec = NumpyArrayCodec()

    def encode(self, item: Image) -> bytes:
        header = struct.pack("<qq", item.timestamp_ns, item.seq)
        image_bytes = self._numpy_codec.encode(item.data)
        return header + image_bytes

    def decode(self, data: bytes | memoryview) -> Image:
        # Ensure we have bytes for struct.unpack
        if isinstance(data, memoryview):
            header_data = bytes(data[:16])
            image_data = data[16:]
        else:
            header_data = data[:16]
            image_data = data[16:]

        timestamp_ns, seq = struct.unpack("<qq", header_data)
        img_data = self._numpy_codec.decode(image_data)

        return Image(
            timestamp_ns=timestamp_ns,
            seq=seq,
            data=img_data,
        )
