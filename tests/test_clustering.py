import geopandas as gpd
from shapely.geometry import LineString, Point

from sigma_engine.clustering import ClusteringConfig, cluster_by_type, retained_points
from sigma_engine.network import RoadNetwork


def test_network_hdbscan_separates_two_dense_groups_on_one_road_component():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    xs = [1, 2, 3, 4, 5, 95, 96, 97, 98, 99]
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(len(xs))],
            "type": ["A"] * len(xs),
        },
        geometry=[Point(x, 0) for x in xs],
        crs=roads.crs,
    )

    base = RoadNetwork.from_geodataframe(roads)
    augmented = base.augment(base.snap(points))
    clustered = cluster_by_type(
        points,
        augmented,
        ClusteringConfig(min_cluster_size=3, min_samples=2),
    )
    retained = retained_points(clustered)

    assert len(retained) == len(points)
    assert set(retained["cluster"]) == {0, 1}
    assert retained.groupby("cluster").size().sort_values().tolist() == [5, 5]


def test_network_hdbscan_is_invariant_to_input_row_order():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    xs = [1, 2, 3, 4, 5, 95, 96, 97, 98, 99]
    original = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(len(xs))],
            "type": ["A"] * len(xs),
        },
        geometry=[Point(x, 0) for x in xs],
        crs=roads.crs,
    )
    reversed_points = original.iloc[::-1].reset_index(drop=True)
    config = ClusteringConfig(min_cluster_size=3, min_samples=2)

    base1 = RoadNetwork.from_geodataframe(roads)
    clustered1 = cluster_by_type(
        original,
        base1.augment(base1.snap(original)),
        config,
    )
    base2 = RoadNetwork.from_geodataframe(roads)
    clustered2 = cluster_by_type(
        reversed_points,
        base2.augment(base2.snap(reversed_points)),
        config,
    )

    labels1 = dict(zip(clustered1["canonical_id"], clustered1["cluster"], strict=True))
    labels2 = dict(zip(clustered2["canonical_id"], clustered2["cluster"], strict=True))
    assert labels1 == labels2


def test_sparse_hdbscan_has_no_5000_point_dense_matrix_guard():
    from sigma_engine.clustering import prepare_sparse_context

    n = 5_101
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (n * 10 + 10, 0)])],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i:05d}" for i in range(n)],
            "type": ["A"] * n,
        },
        geometry=[Point(i * 10 + 1, 0) for i in range(n)],
        crs=roads.crs,
    )
    context = prepare_sparse_context(roads, points, vertex_digits=9)
    clustered = cluster_by_type(
        points,
        context,
        ClusteringConfig(
            min_cluster_size=5,
            min_samples=5,
            max_distance=1.0,
            distance_mode="fixed",
        ),
    )
    assert len(clustered) == n
    assert (clustered["cluster"] == -1).all()


def test_sparse_config_rejects_one_adaptive_distance_step():
    import pytest

    with pytest.raises(ValueError, match="distance_steps must be an integer >= 2"):
        ClusteringConfig(distance_steps=1).validate()


def test_sparse_clustering_records_distance_summary_and_trace():
    from sigma_engine.clustering import prepare_sparse_context

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    xs = [1, 2, 3, 4, 5, 95, 96, 97, 98, 99]
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(len(xs))],
            "type": ["A"] * len(xs),
        },
        geometry=[Point(x, 0) for x in xs],
        crs=roads.crs,
    )
    context = prepare_sparse_context(roads, points)
    clustered = cluster_by_type(
        points,
        context,
        ClusteringConfig(
            min_cluster_size=3,
            min_samples=2,
            max_distance=100.0,
            distance_mode="adaptive",
            min_distance=5.0,
        ),
    )
    summary = clustered.attrs["clustering_summary"]
    trace = clustered.attrs["clustering_trace"]
    assert len(summary) == 1
    assert summary[0]["type"] == "A"
    assert summary[0]["distance_status"] in {"converged", "ceiling_reached"}
    assert summary[0]["distance_steps_run"] >= 1
    assert summary[0]["max_distance"] <= 100.0
    assert trace
    assert {"type", "trial", "radius", "outcome", "n_pairs"}.issubset(trace[0])


def test_sparse_hdbscan_rejects_malformed_sparse_metric_and_fractional_weights():
    import pytest
    from scipy.sparse import coo_matrix

    from sigma_engine._sparse_hdbscan import hdbscan

    asymmetric = coo_matrix(([1.0], ([0], [1])), shape=(2, 2))
    with pytest.raises(ValueError, match="symmetric"):
        hdbscan(asymmetric, min_cluster_size=2, min_samples=1)

    symmetric = coo_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="integer"):
        hdbscan(symmetric, weights=[1.5, 1.0], min_cluster_size=2, min_samples=1)


def test_clustering_can_skip_one_failed_type_and_continue(monkeypatch):
    import sigma_engine.clustering as module

    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)])],
        crs="EPSG:3857",
    )
    xs = [1, 2, 3, 4, 5, 95, 96, 97, 98, 99]
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(len(xs))],
            "type": ["A"] * 5 + ["B"] * 5,
        },
        geometry=[Point(x, 0) for x in xs],
        crs=roads.crs,
    )
    context = module.prepare_sparse_context(roads, points)
    original = module._run_sparse_hdbscan

    def fail_a(graph, snaps, config, *, label, progress):
        if "'A'" in label:
            raise RuntimeError("synthetic clustering failure")
        return original(graph, snaps, config, label=label, progress=progress)

    monkeypatch.setattr(module, "_run_sparse_hdbscan", fail_a)
    events = []
    result = module.cluster_by_type(
        points,
        context,
        ClusteringConfig(
            min_cluster_size=2,
            min_samples=2,
            max_distance=20.0,
            distance_mode="fixed",
            allow_single_cluster=True,
        ),
        continue_on_error=True,
        events=events,
    )

    assert (result.loc[result["type"] == "A", "cluster"] >= 0).any()
    assert (result.loc[result["type"] == "B", "cluster"] >= 0).any()
    assert any(
        e["stage"] == "clustering"
        and e["type"] == "A"
        and e["status"] == "recovered"
        and e["action"] == "fallback_euclidean_hdbscan"
        for e in events
    )
    summary = {row["type"]: row for row in result.attrs["clustering_summary"]}
    assert summary["A"]["clustering_method"] == "euclidean_hdbscan_fallback"


def test_clustering_skips_type_when_network_and_euclidean_hdbscan_both_fail(monkeypatch):
    import sigma_engine.clustering as module

    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs="EPSG:3857")
    points = gpd.GeoDataFrame(
        {"canonical_id": [f"p{i}" for i in range(5)], "type": ["A"] * 5},
        geometry=[Point(x, 0) for x in [1, 2, 3, 4, 5]],
        crs=roads.crs,
    )
    context = module.prepare_sparse_context(roads, points)
    monkeypatch.setattr(
        module, "_run_sparse_hdbscan",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
    )
    monkeypatch.setattr(
        module, "_run_euclidean_hdbscan_by_component",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("euclidean failed")),
    )
    events = []
    result = module.cluster_by_type(
        points, context,
        ClusteringConfig(min_cluster_size=2, min_samples=2, max_distance=20, distance_mode="fixed"),
        continue_on_error=True, events=events,
    )
    assert (result["cluster"] == -1).all()
    assert events[-1]["status"] == "skipped"
    assert events[-1]["action"] == "skip_type_after_euclidean_hdbscan_failure"


def test_euclidean_hdbscan_fallback_never_merges_disconnected_road_components(monkeypatch):
    import sigma_engine.clustering as module

    roads = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (20, 0)]),
            LineString([(0, 1), (20, 1)]),
        ],
        crs="EPSG:3857",
    )
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(6)],
            "type": ["A"] * 6,
        },
        geometry=[Point(1, 0), Point(2, 0), Point(3, 0), Point(1, 1), Point(2, 1), Point(3, 1)],
        crs=roads.crs,
    )
    context = module.prepare_sparse_context(roads, points)
    monkeypatch.setattr(
        module,
        "_run_sparse_hdbscan",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network failed")),
    )
    result = module.cluster_by_type(
        points,
        context,
        ClusteringConfig(
            min_cluster_size=2,
            min_samples=2,
            max_distance=10,
            distance_mode="fixed",
            allow_single_cluster=True,
        ),
        continue_on_error=True,
        events=[],
    )
    retained = result.loc[result["cluster"] >= 0]
    assert retained["cluster"].nunique() == 2
    assert retained.groupby("cluster")["network_component"].nunique().eq(1).all()
