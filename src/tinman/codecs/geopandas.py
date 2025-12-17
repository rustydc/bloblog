"""GeoPandas GeoDataFrame codec using GeoParquet format.

Provides efficient storage for geospatial data with full geometry and CRS support.
"""

from __future__ import annotations

from collections.abc import Buffer
from typing import TYPE_CHECKING, Literal

from ..oblog import Codec

if TYPE_CHECKING:
    import geopandas as gpd

CompressionType = Literal["snappy", "gzip", "brotli", None]


class GeoDataFrameCodec(Codec["gpd.GeoDataFrame"]):
    """Codec for GeoPandas GeoDataFrames using GeoParquet format.

    Stores GeoDataFrames in GeoParquet format which:
    - Preserves geometry columns (Point, Polygon, LineString, etc.)
    - Preserves CRS (Coordinate Reference System)
    - Uses efficient columnar storage with compression
    - Is compatible with other GIS tools

    Note: This is NOT zero-copy due to Parquet decompression.
    For zero-copy numeric data, use DataFrameCodec with regular DataFrames.

    Example:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> from pathlib import Path
        >>> from tinman import ObLog
        >>> from tinman.codecs import GeoDataFrameCodec
        >>>
        >>> # Create a GeoDataFrame
        >>> gdf = gpd.GeoDataFrame({
        ...     'name': ['A', 'B', 'C'],
        ...     'value': [1.0, 2.0, 3.0],
        ...     'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        ... }, crs="EPSG:4326")
        >>>
        >>> # Write GeoDataFrames
        >>> oblog = ObLog(Path("logs"))
        >>> write = oblog.get_writer("locations", GeoDataFrameCodec())
        >>> write(gdf)
        >>> await oblog.close()
        >>>
        >>> # Read GeoDataFrames (CRS and geometry preserved!)
        >>> oblog = ObLog(Path("logs"))
        >>> async for timestamp, gdf in oblog.read_channel("locations"):
        ...     print(gdf.crs)  # EPSG:4326
        ...     print(gdf.geometry)  # Geometry column intact
        >>> await oblog.close()

    Requirements:
        pip install geopandas pyarrow
        # or: pip install tinman[geopandas]
    """

    def __init__(self, compression: CompressionType = "snappy"):
        """Initialize GeoDataFrameCodec.

        Args:
            compression: Parquet compression algorithm.
                Options: 'snappy' (default, fast), 'gzip' (smaller), 'brotli', None
        """
        self.compression: CompressionType = compression

    def encode(self, item: "gpd.GeoDataFrame") -> bytes:
        """Encode GeoDataFrame to GeoParquet bytes.

        Args:
            item: GeoDataFrame to encode

        Returns:
            Bytes containing GeoParquet data with geometry and CRS
        """
        import io

        buffer = io.BytesIO()
        item.to_parquet(buffer, compression=self.compression)
        return buffer.getvalue()

    def decode(self, data: Buffer) -> "gpd.GeoDataFrame":
        """Decode GeoParquet bytes to GeoDataFrame.

        Args:
            data: Buffer containing GeoParquet data

        Returns:
            GeoDataFrame with geometry and CRS preserved
        """
        import geopandas as gpd
        import pyarrow.parquet as pq
        import io

        buffer = io.BytesIO(bytes(data))
        table = pq.read_table(buffer)
        return gpd.GeoDataFrame.from_arrow(table)
