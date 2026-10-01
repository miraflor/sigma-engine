import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from sigma_engine.pipeline import EngineConfig, run_engine

pytest.importorskip("pyarrow")


@pytest.mark.parametrize("classification_column", ["io80_code", "io80_map_code"])
def test_end_to_end(tmp_path, classification_column):
    crs = "EPSG:3857"
    roads_path = tmp_path / "roads.gpkg"
    boundary_path = tmp_path / "boundary.gpkg"
    points_path = tmp_path / "points.parquet"
    io_path = tmp_path / "io.csv"
    out = tmp_path / "out"

    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=crs)
    roads.to_file(roads_path, driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(0, -10, 100, 10)], crs=crs).to_file(boundary_path, driver="GPKG")

    xs = [5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 90, 91, 92, 93, 94]
    types = ["01"] * 10 + ["02"] * 10
    pts = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(20)],
            "canonical_name": [f"P{i}" for i in range(20)],
            classification_column: types,
        },
        geometry=[Point(x, 0) for x in xs],
        crs=crs,
    )
    pts.to_parquet(points_path)

    sectors = [f"{i:02d}" for i in range(1, 81)]
    io = pd.DataFrame(0.0, index=sectors, columns=sectors)
    io.loc["01", "02"] = 10.0
    io.loc["02", "01"] = 1.0
    io.to_csv(io_path)

    result = run_engine(
        EngineConfig(
            roads_path=str(roads_path),
            boundary_path=str(boundary_path),
            io_table_override_path=str(io_path),
            points_path=str(points_path),
            output_dir=str(out),
            min_cluster_size=3,
            min_samples=2,
            allow_single_cluster=True,
            voronoi_resolution=5,
        )
    )
    assert set(["canonical_id", "canonical_name", "type", "centrality_score", "geometry"]).issubset(
        result.points.columns
    )
    assert result.points.centrality_score.notna().all()
    assert result.metadata["eigenvector_degenerate_dag"] is True
    assert gpd.read_parquet(out / "sigma_points_centrality.parquet").crs is not None
    metadata = json.loads((out / "sigma_run_metadata.json").read_text())
    assert metadata["classification"] == "io80"
    assert metadata["point_classification_column"] == classification_column
    summary = pd.read_csv(out / "sigma_clustering_summary.csv")
    trace = pd.read_csv(out / "sigma_clustering_distance_trace.csv")
    assert set(summary["type"].astype(str)) == {"1", "2"} or set(summary["type"].astype(str)) == {"01", "02"}
    assert {"distance_status", "max_distance", "n_pairs"}.issubset(summary.columns)
    assert len(trace) >= 2
    assert metadata["hdbscan_clustering_summary_file"] == "sigma_clustering_summary.csv"
    assert metadata["hdbscan_distance_trace_file"] == "sigma_clustering_distance_trace.csv"
    assert metadata["mwas_requested_mode"] == "fast"
    assert metadata["mwas_method"] == "fast_greedy"
    assert metadata["mwas_exact"] is False
    assert metadata["mwas_optimal"] is False
    assert metadata["io_dag_edge_count"] >= 1
    assert metadata["io_dag_weight"] + metadata["io_removed_weight"] == pytest.approx(
        metadata["io_total_weight_source"]
    )
    dag = pd.read_csv(out / "sigma_io_dag.csv")
    assert {"type_from", "type_to", "io_weight"}.issubset(dag.columns)
    assert set(result.output_paths) >= {
        "clustering_summary", "clustering_distance_trace", "io_dag"
    }


def test_end_to_end_skips_failed_voronoi_type_without_aborting(tmp_path, monkeypatch):
    import sigma_engine.voronoi as voronoi_module

    crs = "EPSG:3857"
    roads_path = tmp_path / "roads.gpkg"
    boundary_path = tmp_path / "boundary.gpkg"
    points_path = tmp_path / "points.parquet"
    io_path = tmp_path / "io.csv"
    out = tmp_path / "out"

    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=crs)
    roads.to_file(roads_path, driver="GPKG")
    gpd.GeoDataFrame(geometry=[box(0, -10, 100, 10)], crs=crs).to_file(boundary_path, driver="GPKG")

    xs = [5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 90, 91, 92, 93, 94]
    types = ["01"] * 10 + ["02"] * 10
    gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(20)],
            "canonical_name": [f"P{i}" for i in range(20)],
            "io80_code": types,
        },
        geometry=[Point(x, 0) for x in xs],
        crs=crs,
    ).to_parquet(points_path)

    sectors = [f"{i:02d}" for i in range(1, 81)]
    io = pd.DataFrame(0.0, index=sectors, columns=sectors)
    io.loc["01", "02"] = 10.0
    io.to_csv(io_path)

    original = voronoi_module.surface_partition_for_type

    def fail_type_01(network, type_points, *args, **kwargs):
        if str(type_points["type"].iloc[0]) == "01":
            raise RuntimeError("synthetic unrecoverable type surface failure")
        return original(network, type_points, *args, **kwargs)

    monkeypatch.setattr(voronoi_module, "surface_partition_for_type", fail_type_01)

    result = run_engine(
        EngineConfig(
            roads_path=str(roads_path),
            boundary_path=str(boundary_path),
            io_table_override_path=str(io_path),
            points_path=str(points_path),
            output_dir=str(out),
            min_cluster_size=3,
            min_samples=2,
            allow_single_cluster=True,
            voronoi_resolution=5,
            fail_fast=False,
        )
    )

    # Network Voronoi failure for type 01 should be recovered by the
    # Euclidean Voronoi fallback rather than dropping the type.
    assert set(result.points["type"]) == {"01", "02"}
    assert set(result.centers["type"]) == {"01", "02"}
    assert set(result.partitions["type"]) == {"01", "02"}

    methods = (
        result.partitions.assign(type=result.partitions["type"].astype(str).str.zfill(2))
        .groupby("type")["surface_method"]
        .first()
        .to_dict()
    )
    assert methods["01"] == "euclidean_voronoi_fallback"
    assert methods["02"] == "network_voronoi"

    events = pd.read_csv(out / "sigma_stage_events.csv", dtype={"type": str})
    recovered = events.loc[
        (events["stage"] == "voronoi")
        & (events["status"] == "recovered")
        & (events["action"] == "fallback_euclidean_voronoi")
    ]
    assert set(recovered["type"].astype(str).str.zfill(2)) == {"01"}
    assert result.metadata["recoverable_stage_recovered_count"] >= 1
    assert result.metadata["recoverable_stage_skipped_count"] == 0
    assert result.metadata["point_count_dropped_recoverable_failures"] == 0


def _checkpoint_fixture(tmp_path):
    crs = "EPSG:3857"
    roads_path = tmp_path / "roads_cp.gpkg"
    boundary_path = tmp_path / "boundary_cp.gpkg"
    points_path = tmp_path / "points_cp.parquet"
    io_path = tmp_path / "io_cp.csv"
    out = tmp_path / "out_cp"

    gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=crs).to_file(
        roads_path, driver="GPKG"
    )
    gpd.GeoDataFrame(geometry=[box(0, -10, 100, 10)], crs=crs).to_file(
        boundary_path, driver="GPKG"
    )
    xs = [5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 90, 91, 92, 93, 94]
    types = ["01"] * 10 + ["02"] * 10
    gpd.GeoDataFrame(
        {
            "canonical_id": [f"cp{i}" for i in range(20)],
            "canonical_name": [f"CP{i}" for i in range(20)],
            "io80_code": types,
        },
        geometry=[Point(x, 0) for x in xs],
        crs=crs,
    ).to_parquet(points_path)
    sectors = [f"{i:02d}" for i in range(1, 81)]
    io = pd.DataFrame(0.0, index=sectors, columns=sectors)
    io.loc["01", "02"] = 10.0
    io.to_csv(io_path)
    return EngineConfig(
        roads_path=str(roads_path),
        boundary_path=str(boundary_path),
        io_table_override_path=str(io_path),
        points_path=str(points_path),
        output_dir=str(out),
        min_cluster_size=3,
        min_samples=2,
        allow_single_cluster=True,
        voronoi_resolution=5,
    ), out


def test_resume_after_crash_does_not_repeat_clustering(tmp_path, monkeypatch):
    import sigma_engine.pipeline as pipeline_module

    config, out = _checkpoint_fixture(tmp_path)
    original_center = pipeline_module.cluster_centers
    original_cluster = pipeline_module.cluster_by_type

    def crash_after_clustering(*args, **kwargs):
        raise RuntimeError("synthetic crash after clustering")

    monkeypatch.setattr(pipeline_module, "cluster_centers", crash_after_clustering)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run_engine(config)

    assert (out / "sigma_clustered_points.parquet").exists()
    assert (out / "sigma_clustering_summary.csv").exists()
    manifest = json.loads((out / ".sigma_checkpoints" / "manifest.json").read_text())
    assert "clustering" in manifest["completed_stages"]
    assert "centers" not in manifest["completed_stages"]

    monkeypatch.setattr(pipeline_module, "cluster_centers", original_center)

    def clustering_must_not_run(*args, **kwargs):
        raise AssertionError("clustering was recomputed instead of resumed")

    monkeypatch.setattr(pipeline_module, "cluster_by_type", clustering_must_not_run)
    result = run_engine(config)
    assert len(result.points) > 0
    assert (out / "sigma_network_centers.parquet").exists()
    assert (out / "sigma_points_with_center_distance.parquet").exists()
    assert (out / "sigma_partitions.parquet").exists()
    assert (out / "sigma_points_partitioned.parquet").exists()
    assert (out / "sigma_io_dag.csv").exists()
    assert (out / "sigma_X_nodes.csv").exists()
    assert (out / "sigma_X_edges.csv").exists()


def test_resume_after_center_checkpoint_skips_median_solver(tmp_path, monkeypatch):
    import sigma_engine.pipeline as pipeline_module

    config, out = _checkpoint_fixture(tmp_path)
    original_distance = pipeline_module._attach_center_distances
    original_center = pipeline_module.cluster_centers

    def crash_after_centers(*args, **kwargs):
        raise RuntimeError("synthetic crash after centers")

    monkeypatch.setattr(pipeline_module, "_attach_center_distances", crash_after_centers)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run_engine(config)
    assert (out / "sigma_network_centers.parquet").exists()
    manifest = json.loads((out / ".sigma_checkpoints" / "manifest.json").read_text())
    assert "centers" in manifest["completed_stages"]

    monkeypatch.setattr(pipeline_module, "_attach_center_distances", original_distance)

    def centers_must_not_run(*args, **kwargs):
        raise AssertionError("1-median stage was recomputed instead of resumed")

    monkeypatch.setattr(pipeline_module, "cluster_centers", centers_must_not_run)
    result = run_engine(config)
    assert len(result.centers) > 0


def test_restart_invalidates_checkpoint(tmp_path, monkeypatch):
    import sigma_engine.pipeline as pipeline_module

    config, out = _checkpoint_fixture(tmp_path)
    run_engine(config)
    called = {"cluster": 0}
    original_cluster = pipeline_module.cluster_by_type

    def counted(*args, **kwargs):
        called["cluster"] += 1
        return original_cluster(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "cluster_by_type", counted)
    run_engine(EngineConfig(**{**config.__dict__, "restart": True}))
    assert called["cluster"] == 1
    manifest = json.loads((out / ".sigma_checkpoints" / "manifest.json").read_text())
    assert manifest["final_output_complete"] is True
