"""Step 7 point scoring for the revised SIGMA workflow."""

from __future__ import annotations

from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from .spatial_graph import make_node_id


def tempered_point_scores(
    points: gpd.GeoDataFrame,
    centrality: Mapping[str, float],
    *,
    distance_tempering: float = 0.15,
    distance_column: str = "distance_to_cluster_median",
) -> gpd.GeoDataFrame:
    """Apply cluster-p90 bounded distance tempering.

    q = min(d / max(Q90(d within cluster), 1), 1)
    S = C * (1 - lambda*q)
    """
    if not 0.0 <= float(distance_tempering) <= 1.0:
        raise ValueError("distance_tempering must be between 0 and 1")

    required = {"type", "cluster", distance_column, "geometry"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"point-score input is missing columns: {sorted(missing)}")
    if points.empty:
        raise ValueError("point-score input is empty")

    out = points.copy()
    distance = pd.to_numeric(out[distance_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("point-to-cluster-median distances must be finite and non-negative")
    out[distance_column] = distance

    node_ids = [
        make_node_id(str(t), int(c))
        for t, c in zip(out["type"], out["cluster"], strict=True)
    ]
    missing_nodes = sorted(set(node_ids) - set(centrality))
    if missing_nodes:
        raise ValueError(f"centrality is missing cluster nodes: {missing_nodes[:10]}")

    out["cluster_centrality"] = np.asarray(
        [float(centrality[node]) for node in node_ids], dtype=float
    )
    if not np.isfinite(out["cluster_centrality"].to_numpy(float)).all():
        raise ValueError("cluster centrality contains non-finite values")

    out["cluster_distance_p90"] = out.groupby(
        ["type", "cluster"], sort=False
    )[distance_column].transform(lambda s: float(np.quantile(s.to_numpy(float), 0.90)))
    scale = np.maximum(out["cluster_distance_p90"].to_numpy(float), 1.0)
    q = np.minimum(distance / scale, 1.0)
    out["normalized_cluster_distance"] = q
    out["distance_tempering"] = float(distance_tempering)
    out["sigma_score"] = out["cluster_centrality"].to_numpy(float) * (
        1.0 - float(distance_tempering) * q
    )

    lower = out["cluster_centrality"].to_numpy(float) * (1.0 - float(distance_tempering))
    upper = out["cluster_centrality"].to_numpy(float)
    score = out["sigma_score"].to_numpy(float)
    tolerance = np.maximum(1e-12, np.abs(upper) * 1e-12)
    if np.any(score < lower - tolerance) or np.any(score > upper + tolerance):
        raise RuntimeError("tempered point score violated its configured within-cluster bounds")

    if "point_id" not in out.columns:
        if "canonical_id" in out.columns:
            out["point_id"] = out["canonical_id"]
        else:
            out["point_id"] = out.index.astype(str)

    preferred = [
        "point_id",
        "type",
        "cluster",
        "cluster_centrality",
        distance_column,
        "cluster_distance_p90",
        "normalized_cluster_distance",
        "distance_tempering",
        "sigma_score",
        "geometry",
    ]
    # Preserve canonical_id/name when present because current sigma-engine inputs use them.
    compatibility = [c for c in ["canonical_id", "canonical_name"] if c in out.columns]
    columns = preferred[:-1] + compatibility + ["geometry"]
    return out[columns].copy()
