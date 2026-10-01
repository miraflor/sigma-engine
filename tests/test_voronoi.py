import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from sigma_engine.network import RoadNetwork
from sigma_engine.voronoi import all_surface_partitions, surface_partition_for_type


def _augment_with_clustered_points(roads, points):
    base = RoadNetwork.from_geodataframe(roads)
    augmented = base.augment(base.snap(points))
    points = points.copy()
    points["network_node"] = augmented.point_node
    return augmented, points


def test_surface_partition_has_one_polygon_per_cluster():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -2, 20, 2)], crs=roads.crs)

    result = surface_partition_for_type(
        augmented,
        points,
        boundary,
        resolution=1.0,
        max_cells=1000,
    )

    assert set(result.cluster) == {0, 1}
    assert np.isclose(result.geometry.area.sum(), boundary.geometry.area.sum())


def test_all_partitions_preserve_economic_type_not_geometry_type():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A", "B", "B"], "cluster": [0, 1, 0, 1]},
        geometry=[Point(2, 0), Point(18, 0), Point(4, 0), Point(16, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -2, 20, 2)], crs=roads.crs)

    result = all_surface_partitions(
        augmented,
        points,
        boundary,
        resolution=1.0,
        max_cells=1000,
    )

    assert set(result["type"]) == {"A", "B"}


def test_surface_ignores_source_less_road_components_instead_of_leaving_holes():
    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(90, 0), (100, 0)]),
        ],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A"], "cluster": [0]},
        geometry=[Point(5, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -5, 100, 5)], crs=roads.crs)

    result = surface_partition_for_type(
        augmented,
        points,
        boundary,
        resolution=5.0,
        max_cells=1000,
    )

    assert set(result.cluster) == {0}
    assert np.isclose(result.geometry.area.sum(), boundary.geometry.area.sum())


def test_surface_fails_loudly_when_grid_is_too_coarse_to_represent_a_cluster():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(1, 0), Point(2, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -5, 100, 5)], crs=roads.crs)

    with pytest.raises(ValueError, match="too coarse"):
        surface_partition_for_type(
            augmented,
            points,
            boundary,
            resolution=100.0,
            max_cells=100,
        )


def test_network_voronoi_exact_midpoint_tie_goes_to_smaller_cluster():
    from sigma_engine.voronoi import network_voronoi

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A", "A"], "cluster": [1, 0, 99]},
        geometry=[Point(0, 0), Point(10, 0), Point(5, 0)],
        crs=roads.crs,
    )
    # The midpoint point is only used to insert a graph vertex, not as a Voronoi source.
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    result = network_voronoi(
        aug,
        aug.point_node[:2],
        np.array([1, 0], dtype=np.int64),
    )
    midpoint = int(aug.point_node[2])
    assert result.node_distance[midpoint] == pytest.approx(5.0)
    assert int(result.node_cluster[midpoint]) == 0


def test_all_partitions_auto_refines_grid_when_coarse_cell_erases_cluster():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(1, 0), Point(2, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -5, 100, 5)], crs=roads.crs)
    events = []

    result = all_surface_partitions(
        augmented,
        points,
        boundary,
        resolution=100.0,
        max_cells=1000,
        auto_refine=True,
        refine_factor=0.5,
        max_refinements=6,
        continue_on_error=True,
        events=events,
    )

    assert set(result.cluster) == {0, 1}
    assert result["surface_resolution"].max() < 100.0
    assert any(e["status"] == "recovered" for e in events)


def test_all_partitions_can_skip_one_failed_type_and_continue(monkeypatch):
    import sigma_engine.voronoi as module

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    boundary = gpd.GeoDataFrame(geometry=[box(0, -2, 20, 2)], crs=roads.crs)
    original = module.surface_partition_for_type

    def fail_a(network, type_points, *args, **kwargs):
        if str(type_points["type"].iloc[0]) == "A":
            raise RuntimeError("synthetic per-type failure")
        return original(network, type_points, *args, **kwargs)

    monkeypatch.setattr(module, "surface_partition_for_type", fail_a)
    events = []
    result = module.all_surface_partitions(
        augmented,
        points,
        boundary,
        resolution=1.0,
        max_cells=1000,
        continue_on_error=True,
        events=events,
    )

    assert set(result["type"]) == {"B"}
    assert any(e["stage"] == "voronoi" and e["type"] == "A" and e["status"] == "skipped" for e in events)


def test_network_voronoi_uses_euclidean_fallback_after_six_attempts(monkeypatch):
    import sigma_engine.voronoi as module

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    centers = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(0, -2, 20, 2)], crs=roads.crs)
    attempts = []

    def always_too_coarse(*args, **kwargs):
        attempts.append(float(args[3]))
        raise ValueError("synthetic too coarse network Voronoi")

    monkeypatch.setattr(module, "surface_partition_for_type", always_too_coarse)
    events = []
    result = module.all_surface_partitions(
        augmented,
        points,
        boundary,
        resolution=16.0,
        max_cells=10_000,
        centers=centers,
        auto_refine=True,
        refine_factor=0.5,
        max_refinements=5,
        euclidean_fallback=True,
        continue_on_error=True,
        events=events,
    )

    assert attempts == [16.0, 8.0, 4.0, 2.0, 1.0, 0.5]
    assert set(result["cluster"]) == {0, 1}
    assert set(result["surface_method"]) == {"euclidean_voronoi_fallback"}
    assert result["surface_resolution"].isna().all()
    assert any(e["action"] == "fallback_euclidean_voronoi" for e in events)
    assert not any(e["status"] == "skipped" for e in events)


def test_euclidean_voronoi_duplicate_center_sites_use_raw_cluster_centroids():
    from sigma_engine.voronoi import euclidean_surface_partition_for_type

    points = gpd.GeoDataFrame(
        {"type": ["A", "A", "A", "A"], "cluster": [0, 0, 1, 1]},
        geometry=[Point(1, 0), Point(2, 0), Point(8, 0), Point(9, 0)],
        crs="EPSG:3857",
    )
    centers = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(5, 0), Point(5, 0)],
        crs=points.crs,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(0, -5, 10, 5)], crs=points.crs)

    result = euclidean_surface_partition_for_type(
        points,
        centers,
        boundary,
        points.crs,
    )

    assert set(result["surface_seed_method"]) == {"raw_cluster_centroid_duplicate_center"}
    assert np.isclose(result.geometry.area.sum(), boundary.geometry.area.sum())


def test_euclidean_voronoi_failure_is_only_then_skipped(monkeypatch):
    import sigma_engine.voronoi as module

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    augmented, points = _augment_with_clustered_points(roads, points)
    centers = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 0]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    boundary = gpd.GeoDataFrame(geometry=[box(0, -2, 20, 2)], crs=roads.crs)
    original_network = module.surface_partition_for_type
    original_euclidean = module.euclidean_surface_partition_for_type

    def fail_network_for_a(network, type_points, *args, **kwargs):
        if str(type_points["type"].iloc[0]) == "A":
            raise RuntimeError("synthetic network failure")
        return original_network(network, type_points, *args, **kwargs)

    def fail_euclidean_for_a(type_points, *args, **kwargs):
        if str(type_points["type"].iloc[0]) == "A":
            raise RuntimeError("synthetic Euclidean failure")
        return original_euclidean(type_points, *args, **kwargs)

    monkeypatch.setattr(module, "surface_partition_for_type", fail_network_for_a)
    monkeypatch.setattr(module, "euclidean_surface_partition_for_type", fail_euclidean_for_a)
    events = []
    result = module.all_surface_partitions(
        augmented,
        points,
        boundary,
        resolution=1.0,
        max_cells=1000,
        centers=centers,
        euclidean_fallback=True,
        continue_on_error=True,
        events=events,
    )

    assert set(result["type"]) == {"B"}
    assert any(
        e["stage"] == "voronoi"
        and e["type"] == "A"
        and e["status"] == "skipped"
        and e["action"] == "skip_type_after_euclidean_voronoi_failure"
        for e in events
    )
