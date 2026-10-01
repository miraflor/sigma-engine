"""Normalize classified point products into SIGMA Engine's narrow input contract.

SIGMA Engine is intentionally producer-agnostic.  It does not launch or import a POI
acquisition/classification application; it accepts a classified vector file and normalizes
only the fields required by the spatial-economic workflow.

Two established column conventions are recognized automatically:

* ``io80_code`` / ``io16_code`` -- used by one compatible producer; and
* ``io80_map_code`` / ``io16_map_code`` -- map-safe, unambiguous fields used by another.

Plural fields such as ``io80_codes`` are intentionally *not* auto-selected because they may
encode a set of candidate sectors rather than one unambiguous sector per point.  A caller may
still name any custom single-valued column explicitly through the CLI/configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import pandas as pd

from .io_utils import normalize_sector_id

Classification = Literal["io80", "io16"]

_AUTO_CLASSIFICATION_COLUMNS: dict[Classification, tuple[str, ...]] = {
    "io80": ("io80_code", "io80_map_code"),
    "io16": ("io16_code", "io16_map_code"),
}


@dataclass(frozen=True)
class PointInputInfo:
    """Describe how one producer file was normalized for the engine."""

    classification_column: str
    canonical_name_source: str


def _normalized_series(series: pd.Series) -> pd.Series:
    """Normalize a candidate sector column without mutating the source frame."""
    return series.map(normalize_sector_id)


def _choose_classification_column(
    frame: gpd.GeoDataFrame,
    classification: Classification,
    explicit_column: str | None,
) -> str:
    """Resolve one single-valued sector column, rejecting ambiguous auto-detection.

    When more than one recognized convention is present, the columns may coexist only if
    every row classified by both agrees after sector-code normalization.  If they agree, the
    column with the most classified rows is preferred; ties follow the stable convention
    order in ``_AUTO_CLASSIFICATION_COLUMNS``.
    """
    if explicit_column is not None:
        column = explicit_column.strip()
        if not column:
            raise ValueError("explicit classification column must not be blank")
        if column not in frame.columns:
            raise ValueError(f"selected classification column is missing: {column}")
        return column

    candidates = [
        column
        for column in _AUTO_CLASSIFICATION_COLUMNS[classification]
        if column in frame.columns
    ]
    if not candidates:
        expected = ", ".join(_AUTO_CLASSIFICATION_COLUMNS[classification])
        raise ValueError(
            f"could not auto-detect a {classification} classification column; "
            f"expected one of: {expected}. Use --{classification}-column for a custom field."
        )

    normalized = {column: _normalized_series(frame[column]) for column in candidates}
    if len(candidates) > 1:
        first = candidates[0]
        for other in candidates[1:]:
            both = normalized[first].notna() & normalized[other].notna()
            conflict = both & normalized[first].ne(normalized[other])
            if bool(conflict.any()):
                sample_rows = frame.index[conflict].tolist()[:10]
                raise ValueError(
                    f"recognized {classification} columns {first!r} and {other!r} disagree "
                    f"on rows {sample_rows}; select the intended field explicitly with "
                    f"--{classification}-column"
                )

    # Prefer the field that actually classifies the greatest number of points.  ``max`` is
    # stable here because candidates already follow a deterministic priority order.
    return max(candidates, key=lambda column: int(normalized[column].notna().sum()))


def _ensure_canonical_name(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    """Ensure ``canonical_name`` exists without making it an upstream hard requirement.

    A compact GIS export may intentionally omit names.  SIGMA Engine needs a stable display
    field in its final artifact but does not use names in any algorithm, so the conservative
    fallback is: ``canonical_name`` -> ``name`` -> textual ``canonical_id``.
    """
    out = frame.copy()
    if "canonical_name" in out.columns:
        source = "canonical_name"
    elif "name" in out.columns:
        out["canonical_name"] = out["name"]
        source = "name"
    else:
        out["canonical_name"] = out["canonical_id"].astype(str)
        source = "canonical_id"

    # Missing/blank names are harmless for the algorithms but poor final metadata.  Fill
    # only those rows from the immutable point identifier rather than dropping the points.
    text = out["canonical_name"].astype("string")
    missing = text.isna() | text.str.strip().eq("")
    if bool(missing.any()):
        out.loc[missing, "canonical_name"] = out.loc[missing, "canonical_id"].astype(str)
        source = f"{source}+canonical_id-fallback"
    return out, source


def prepare_point_input(
    frame: gpd.GeoDataFrame,
    *,
    classification: Classification,
    io80_column: str | None = None,
    io16_column: str | None = None,
) -> tuple[gpd.GeoDataFrame, PointInputInfo]:
    """Return a validated, classified point frame plus normalization metadata.

    The returned frame preserves upstream columns and geometry, adds/normalizes ``type``,
    and guarantees ``canonical_id`` and ``canonical_name``.  Unclassified rows are removed;
    HDBSCAN/noise filtering happens later in the engine pipeline.
    """
    if classification not in _AUTO_CLASSIFICATION_COLUMNS:
        raise ValueError("classification must be 'io80' or 'io16'")
    if frame.crs is None:
        raise ValueError("point CRS is missing")

    required = {"canonical_id", "geometry"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"point data are missing required columns: {sorted(missing)}")

    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError("point geometry contains missing or empty values")
    non_points = ~frame.geometry.geom_type.eq("Point")
    if bool(non_points.any()):
        kinds = sorted(set(frame.loc[non_points].geometry.geom_type.astype(str)))
        raise ValueError(f"point input must contain only Point geometry; found {kinds}")

    explicit = io80_column if classification == "io80" else io16_column
    classification_column = _choose_classification_column(frame, classification, explicit)

    out, name_source = _ensure_canonical_name(frame)
    out["type"] = _normalized_series(out[classification_column])

    # Unclassified records cannot participate in type-wise clustering or the IO graph.
    out = out.loc[out["type"].notna()].copy()
    if out.empty:
        raise ValueError("no points have the selected economic classification")

    if out["canonical_id"].isna().any():
        raise ValueError("canonical_id contains missing values")
    id_text = out["canonical_id"].astype(str)
    if id_text.str.strip().eq("").any():
        raise ValueError("canonical_id contains blank values")
    if id_text.duplicated().any():
        duplicates = id_text.loc[id_text.duplicated(keep=False)].unique()
        raise ValueError(
            "canonical_id must be unique after text normalization; duplicates include "
            f"{duplicates[:10].tolist()}"
        )

    return (
        out.reset_index(drop=True),
        PointInputInfo(
            classification_column=classification_column,
            canonical_name_source=name_source,
        ),
    )
