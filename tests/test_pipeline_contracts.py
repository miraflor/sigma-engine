import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from sigma_engine.pipeline import EngineConfig, run_engine


def _write_point_gpkg(path, sector="A", x=0.0, y=0.0):
    frame = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1"],
            "canonical_name": ["Point 1"],
            "io80_code": [sector],
        },
        geometry=[Point(x, y)],
        crs="EPSG:3857",
    )
    frame.to_file(path, driver="GPKG")


def test_pipeline_rejects_point_sector_missing_from_io_before_network_work(tmp_path):
    points_path = tmp_path / "points.gpkg"
    _write_point_gpkg(points_path, sector="B")

    # These files only need to exist for the early configuration check; type compatibility
    # is intentionally validated before road/boundary parsing and all expensive network work.
    roads_path = tmp_path / "roads.gpkg"
    boundary_path = tmp_path / "boundary.gpkg"
    roads_path.touch()
    boundary_path.touch()

    with pytest.raises(ValueError, match="absent from the IO table"):
        run_engine(
            EngineConfig(
                roads_path=str(roads_path),
                boundary_path=str(boundary_path),
                points_path=str(points_path),
                output_dir=str(tmp_path / "out"),
            )
        )


def test_fail_soft_partition_alignment_drops_only_missing_cluster_nodes():
    from sigma_engine.pipeline import _align_to_successful_partitions
    from sigma_engine.spatial_graph import make_node_id

    crs = "EPSG:3857"
    keep = make_node_id("02", 0)
    drop = make_node_id("01", 0)
    retained = gpd.GeoDataFrame(
        {"node_id": [drop, keep], "type": ["01", "02"], "cluster": [0, 0]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs=crs,
    )
    centers = retained.copy()
    partitions = gpd.GeoDataFrame(
        {"node_id": [keep], "type": ["02"], "cluster": [0]},
        geometry=[Point(1, 0).buffer(1)],
        crs=crs,
    )

    kept_points, kept_centers, missing = _align_to_successful_partitions(
        retained, centers, partitions, fail_fast=False
    )

    assert kept_points["node_id"].tolist() == [keep]
    assert kept_centers["node_id"].tolist() == [keep]
    assert missing == [drop]


def test_point_center_distance_uses_euclidean_fallback_per_cluster(monkeypatch):
    import sigma_engine.pipeline as module
    from sigma_engine.network import RoadNetwork

    from shapely.geometry import LineString
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 0]},
        geometry=[Point(2, 0), Point(4, 0)],
        crs=roads.crs,
    )
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    points["network_node"] = aug.point_node
    center_node = int(aug.point_node[0])
    centers = gpd.GeoDataFrame(
        {
            "type": ["A"],
            "cluster": [0],
            "center_network_node": [center_node],
        },
        geometry=[aug.nodes.geometry.iloc[center_node]],
        crs=roads.crs,
    )

    monkeypatch.setattr(
        module.SparseRoadSolver,
        "point_distances_from_centers",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network distance failed")),
    )
    events = []
    out = module._attach_center_distances(
        points,
        centers,
        aug,
        continue_on_error=True,
        events=events,
    )
    assert out["point_center_distance_method"].eq("euclidean_fallback").all()
    assert out["network_distance_to_center"].tolist() == pytest.approx([0.0, 2.0])
    assert events[-1]["action"] == "fallback_euclidean_point_center_distance"
    assert events[-1]["status"] == "recovered"
