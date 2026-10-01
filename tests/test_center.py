import geopandas as gpd
from shapely.geometry import LineString, Point

from sigma_engine.center import network_1_median
from sigma_engine.network import RoadNetwork


def test_network_median_on_line():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(geometry=[Point(2, 0), Point(3, 0), Point(18, 0)], crs=roads.crs)
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    med = network_1_median(aug, aug.point_node)
    assert med.geometry.equals(Point(3, 0))


def test_network_median_compresses_duplicate_demand_exactly():
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (30, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        geometry=[Point(2, 0), Point(2, 0), Point(2, 0), Point(25, 0)], crs=roads.crs
    )
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    med = network_1_median(aug, aug.point_node)
    assert med.geometry.equals(Point(2, 0))


def test_cluster_centers_can_skip_one_failed_cluster_and_continue(monkeypatch):
    import sigma_engine.center as module

    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 1]},
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    points["network_node"] = aug.point_node
    original = module.network_1_median
    calls = {"n": 0}

    def fail_first(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic center failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "network_1_median", fail_first)
    events = []
    result = module.cluster_centers(
        points,
        aug,
        continue_on_error=True,
        events=events,
    )

    assert len(result) == 2
    recovered = result.set_index("cluster").loc[0]
    assert recovered["center_method"] == "euclidean_centroid_road_snap_fallback"
    assert any(
        e["stage"] == "center"
        and e["cluster"] == 0
        and e["status"] == "recovered"
        and e["action"] == "fallback_euclidean_centroid_road_snap"
        for e in events
    )


def test_cluster_centers_skip_when_network_and_centroid_fallback_both_fail(monkeypatch):
    import sigma_engine.center as module

    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"type": ["A"], "cluster": [0]}, geometry=[Point(2, 0)], crs=roads.crs
    )
    base = RoadNetwork.from_geodataframe(roads)
    aug = base.augment(base.snap(points))
    points["network_node"] = aug.point_node
    monkeypatch.setattr(
        module, "network_1_median",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
    )
    monkeypatch.setattr(
        module, "_euclidean_centroid_snap_to_network_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("centroid failed")),
    )
    events = []
    result = module.cluster_centers(points, aug, continue_on_error=True, events=events)
    assert result.empty
    assert events[-1]["status"] == "skipped"
    assert events[-1]["action"] == "skip_cluster_after_centroid_fallback_failure"
