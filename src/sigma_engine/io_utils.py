"""Small I/O helpers shared across the SIGMA engine.

This module deliberately keeps file-format concerns and sector-code normalization out of
algorithm modules.  Doing that gives the rest of the package one simple contract:

* vector inputs arrive as :class:`geopandas.GeoDataFrame` objects; and
* economic sector identifiers are either a normalized string or ``None``.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Spreadsheet programs commonly coerce codes such as ``01`` to ``1`` or ``1.0``.
# SIGMA's IO16/IO80 identifiers are integer-like, so accept only an integer with an
# optional all-zero decimal suffix.  Textual labels (for synthetic/test IO tables) pass
# through untouched.
_NUMERIC_CODE = re.compile(r"^([+-]?\d+)(?:\.0+)?$")


def normalize_sector_id(value: object) -> str | None:
    """Return SIGMA's canonical string representation of one sector identifier.

    Numeric spreadsheet representations are normalized without going through binary
    floating point.  Avoiding ``float(value)`` matters because it prevents accidental
    precision loss for long identifiers, even though today's IO16/IO80 codes are short.

    Examples
    --------
    ``1``, ``1.0``, ``"1"`` and ``"01"`` all become ``"01"``.  Non-numeric labels
    such as ``"A"`` are stripped but otherwise preserved.  Missing/blank values become
    ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = _NUMERIC_CODE.fullmatch(text)
    if match is None:
        return text

    integer = int(match.group(1))
    # ``02d`` means "at least two digits"; it does not truncate larger identifiers.
    return f"{integer:02d}"


def read_vector(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector dataset supported by GeoPandas.

    GeoParquet is handled explicitly because ``geopandas.read_file`` does not read it.
    All other formats (GeoPackage, Shapefile, etc.) are delegated to GeoPandas/Fiona or
    Pyogrio.  ``layer`` is relevant only to container formats such as GeoPackage.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"vector input does not exist: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path, layer=layer)


def write_geoparquet(frame: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Write and immediately validate a GeoParquet artifact.

    The read-back is intentional.  A successful ``to_parquet`` call alone does not prove
    that the file retained geospatial metadata in a form GeoPandas/QGIS can consume.  The
    small extra I/O cost buys an early, local failure instead of a bad final deliverable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frame.crs is None:
        raise ValueError("GeoParquet output requires a CRS")
    if "geometry" not in frame.columns:
        raise ValueError("GeoParquet output requires an active geometry column")

    frame.to_parquet(path, index=False)

    # Round-trip through GeoPandas to verify both row count and CRS metadata.  Geometry
    # validity is *not* universally required (e.g. some workflows intentionally carry
    # invalid source geometry), so we do not silently repair or reject it here.
    check = gpd.read_parquet(path)
    if check.crs is None:
        raise RuntimeError(f"GeoParquet round-trip lost CRS metadata: {path}")
    if len(check) != len(frame):
        raise RuntimeError(
            f"GeoParquet round-trip changed row count for {path}: "
            f"expected {len(frame)}, got {len(check)}"
        )
    return path
