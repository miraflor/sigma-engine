from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from sigma_engine.pipeline import (
    _assert_cluster_invariant,
    _normalize_restart_points,
    _public_clustered,
)


def test_cluster_key_invariant_requires_exact_match():
    points = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0], "distance_to_cluster_median": [1.0, 2.0]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    )
    centers = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0], "center_network_node": [1, 2]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs=points.crs,
    )
    partitions = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0]},
        geometry=[box(-1, -1, 1, 1), box(0, -1, 2, 1)],
        crs=points.crs,
    )
    assert _assert_cluster_invariant(points, centers, partitions) == 2

    bad = partitions.iloc[:1].copy()
    with pytest.raises(RuntimeError, match="cluster-count/key invariant"):
        _assert_cluster_invariant(points, centers, bad)


def test_restart_accepts_legacy_distance_field_only_as_input_alias():
    old = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1"],
            "type": ["A"],
            "cluster": [0],
            "network_distance_to_center": [12.5],
            "point_center_distance_method": ["network_shortest_path"],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:3857",
    )
    new = _normalize_restart_points(old)
    assert new.loc[0, "distance_to_cluster_median"] == 12.5


def test_restart_filters_only_legacy_extra_point_clusters():
    from sigma_engine.pipeline import _align_legacy_restart_points

    points = gpd.GeoDataFrame(
        {
            "type": ["A", "B", "C"],
            "cluster": [0, 0, 0],
            "distance_to_cluster_median": [1.0, 2.0, 3.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs="EPSG:3857",
    )
    centers = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0], "center_network_node": [1, 2]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs=points.crs,
    )
    partitions = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0]},
        geometry=[box(-1, -1, 1, 1), box(0, -1, 2, 1)],
        crs=points.crs,
    )
    aligned, dropped = _align_legacy_restart_points(points, centers, partitions)
    assert dropped == 1
    assert set(aligned["type"]) == {"A", "B"}


def test_restart_rejects_legacy_euclidean_distance_fallback():
    old = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1"],
            "type": ["A"],
            "cluster": [0],
            "network_distance_to_center": [12.5],
            "point_center_distance_method": ["euclidean_fallback"],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="non-network distance method"):
        _normalize_restart_points(old)


def test_restart_rejects_duplicate_point_identifiers():
    saved = gpd.GeoDataFrame(
        {
            "point_id": ["p1", "p1"],
            "type": ["A", "A"],
            "cluster": [0, 0],
            "distance_to_cluster_median": [1.0, 2.0],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    )
    with pytest.raises(ValueError, match="point_id must be non-blank and unique"):
        _normalize_restart_points(saved)


def test_public_clustered_includes_step1_network_position():
    clustered = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1", "p2"],
            "type": ["01", "01"],
            "cluster": [0, -1],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    )
    snaps = SimpleNamespace(
        snapped_xy=np.asarray([[0.2, 0.0], [0.8, 0.0]], dtype=float),
        snap_distance=np.asarray([0.2, 0.2], dtype=float),
    )
    out = _public_clustered(clustered, SimpleNamespace(snaps=snaps))
    assert out["point_id"].tolist() == ["p1", "p2"]
    assert out["snap_distance"].tolist() == [0.2, 0.2]
    assert [geometry.x for geometry in out["network_position"]] == [0.2, 0.8]
