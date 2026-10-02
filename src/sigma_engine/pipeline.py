"""Revised SIGMA v0.1.0 workflow orchestration.

The public stage artifacts are deliberate restart boundaries.  In particular,
``run_from_partitions`` consumes the Step-3/4 artifacts and executes Steps 5--7
without recomputing clustering, medians, point-center distances, or partitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from ._sparse_solver import SparseRoadSolver
from ._version import __version__
from .builtin_io import load_builtin_total_output, load_engine_io_table
from .center import cluster_centers
from .clustering import ClusteringConfig, cluster_by_type, prepare_sparse_context, retained_points
from .io_utils import normalize_sector_id, read_vector, write_geoparquet
from .io_workflow import (
    build_io_dag,
    derive_technical_coefficients,
    read_technical_coefficients,
    read_total_output_vector,
)
from .network import AugmentedNetwork, RoadNetwork, SnappedPoints
from .point_input import prepare_point_input
from .scoring import tempered_point_scores
from .spatial_graph import directed_eigenvector_centrality, instantiate_network, make_node_id
from .voronoi import all_surface_partitions

Classification = Literal["io80", "io16"]
MWASMethod = Literal["fast", "exact"]
CentralityDirection = Literal["incoming", "outgoing"]


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for Steps 1--7 from classified points."""

    roads_path: str
    boundary_path: str
    output_dir: str
    points_path: str

    technical_coefficients_path: str | None = None
    classification: Classification = "io80"
    transactions_override_path: str | None = None
    io80_column: str | None = None
    io16_column: str | None = None
    roads_layer: str | None = None
    boundary_layer: str | None = None
    transactions_sheet: str | int = 0
    technical_coefficients_sheet: str | int = 0
    mwas_method: MWASMethod = "fast"

    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    allow_single_cluster: bool = False
    hdbscan_max_distance: float = 5_000.0
    hdbscan_distance_mode: Literal["adaptive", "fixed"] = "adaptive"
    hdbscan_min_distance: float | None = None
    hdbscan_distance_growth: float = 1.5
    hdbscan_distance_steps: int = 4
    hdbscan_core_truncation_tolerance: float = 0.01
    hdbscan_stability_tolerance: float = 1e-12
    hdbscan_max_neighbor_pairs: int = 20_000_000

    vertex_digits: int = 11
    max_snap_distance: float | None = None
    voronoi_resolution: float = 500.0
    voronoi_max_cells: int = 1_000_000
    voronoi_refine_factor: float = 0.5
    voronoi_max_refinements: int = 5

    centrality_direction: CentralityDirection = "incoming"
    distance_tempering: float = 0.15
    progress: bool = False


@dataclass(frozen=True)
class ContinueConfig:
    """Configuration for executing Steps 5--7 from existing Step-3/4 artifacts."""

    roads_path: str
    partitions_path: str
    output_dir: str

    technical_coefficients_path: str | None = None
    centers_path: str | None = None
    points_with_center_distance_path: str | None = None
    classification: Classification = "io80"
    transactions_override_path: str | None = None
    roads_layer: str | None = None
    transactions_sheet: str | int = 0
    technical_coefficients_sheet: str | int = 0
    mwas_method: MWASMethod = "fast"
    vertex_digits: int = 11
    centrality_direction: CentralityDirection = "incoming"
    distance_tempering: float = 0.15
    progress: bool = False


@dataclass(frozen=True)
class EngineResult:
    """Primary in-memory artifacts and written paths."""

    points: gpd.GeoDataFrame
    centers: gpd.GeoDataFrame
    partitions: gpd.GeoDataFrame
    output_paths: dict[str, str]
    metadata: dict[str, object]


def _report(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[sigma-engine] {message}", flush=True)


def _require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _expected_sector_count(classification: str) -> int:
    if classification == "io80":
        return 80
    if classification == "io16":
        return 16
    raise ValueError("classification must be 'io80' or 'io16'")


def _validate_common(
    *,
    classification: str,
    mwas_method: str,
    centrality_direction: str,
    distance_tempering: float,
) -> None:
    _expected_sector_count(classification)
    if mwas_method not in {"fast", "exact"}:
        raise ValueError("mwas_method must be 'fast' or 'exact'")
    if centrality_direction not in {"incoming", "outgoing"}:
        raise ValueError("centrality_direction must be 'incoming' or 'outgoing'")
    if not 0.0 <= float(distance_tempering) <= 1.0:
        raise ValueError("distance_tempering must be between 0 and 1")


def _validate_point_types(points: gpd.GeoDataFrame, transactions: pd.DataFrame) -> None:
    spatial = {_normalize_type_value(value) for value in points["type"]}
    economic = {_normalize_type_value(value) for value in transactions.index}
    missing = sorted(spatial - economic)
    if missing:
        raise ValueError(
            "classified point sectors are absent from the transaction matrix: "
            f"{missing[:20]}"
        )


def _normalize_type_value(value: object) -> str:
    normalized = normalize_sector_id(value)
    if normalized is None:
        raise ValueError("economic type identifiers must be non-blank")
    return normalized


def _normalize_type_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "type" not in out.columns:
        raise ValueError("artifact is missing required type column")
    out["type"] = [_normalize_type_value(value) for value in out["type"]]
    return out


def _node_keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(
        zip(
            (_normalize_type_value(value) for value in frame["type"]),
            frame["cluster"].astype(int),
            strict=True,
        )
    )


def _assert_cluster_invariant(
    points: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
) -> int:
    point_keys = _node_keys(points)
    center_keys = _node_keys(centers)
    partition_keys = _node_keys(partitions)
    if point_keys != center_keys or center_keys != partition_keys:
        raise RuntimeError(
            "cluster-count/key invariant failed: retained point clusters, network centers, "
            "and partitions must match exactly; "
            f"points-only={sorted(point_keys-center_keys)[:10]}, "
            f"centers-only={sorted(center_keys-point_keys)[:10]}, "
            f"missing-partitions={sorted(center_keys-partition_keys)[:10]}, "
            f"extra-partitions={sorted(partition_keys-center_keys)[:10]}"
        )
    if len(centers) != len(center_keys):
        raise RuntimeError("network centers must have exactly one row per [type, cluster]")
    if len(partitions) != len(partition_keys):
        raise RuntimeError("partitions must have exactly one row per [type, cluster]")
    return len(center_keys)


def _public_clustered(clustered: gpd.GeoDataFrame, sparse_context) -> gpd.GeoDataFrame:
    """Return the public Step-2 artifact including its snapped network position."""
    out = clustered.drop(columns=["_sparse_source_pos"], errors="ignore").copy()
    if len(out) != len(sparse_context.snaps.snapped_xy):
        raise RuntimeError("clustered rows and Step-1 network positions are misaligned")
    if "point_id" not in out.columns:
        if "canonical_id" not in out.columns:
            raise RuntimeError("clustered output is missing a stable point identifier")
        out["point_id"] = out["canonical_id"]
    out["network_position"] = gpd.GeoSeries(
        shapely.points(sparse_context.snaps.snapped_xy),
        index=out.index,
        crs=out.crs,
    )
    if "snap_distance" not in out.columns:
        out["snap_distance"] = sparse_context.snaps.snap_distance.astype(float)
    return out


def _build_augmented_for_retained(
    retained: gpd.GeoDataFrame,
    sparse_context,
    roads_crs,
    *,
    vertex_digits: int,
    progress: bool,
) -> tuple[gpd.GeoDataFrame, AugmentedNetwork]:
    """Reuse the exact Step-1 snap positions selected by sparse HDBSCAN."""
    if "_sparse_source_pos" not in retained.columns:
        raise RuntimeError("cluster output is missing its internal sparse source position")
    source_pos = retained["_sparse_source_pos"].to_numpy(np.int64, copy=False)
    roads = RoadNetwork.from_sparse_graph(
        sparse_context.graph,
        roads_crs,
        vertex_digits=vertex_digits,
    )
    sparse_snaps = sparse_context.snaps.subset(source_pos)
    snapped = SnappedPoints(
        pd.DataFrame(
            {
                "source_pos": np.arange(len(retained), dtype=np.int64),
                "edge_pos": sparse_context.edge_id[source_pos],
                "edge_id": sparse_context.edge_id[source_pos],
                "offset": sparse_snaps.offset,
                "snap_distance": sparse_snaps.snap_distance,
                "snapped_geometry": list(shapely.points(sparse_snaps.snapped_xy)),
            }
        ),
        roads.crs,
    )
    augmented = roads.augment(
        snapped,
        progress=(lambda m: _report(True, m)) if progress else None,
    )
    out = retained.copy().reset_index(drop=True)
    out["network_node"] = augmented.point_node
    out["snap_distance"] = snapped.frame["snap_distance"].to_numpy(float)
    out["network_position"] = snapped.frame["snapped_geometry"].to_numpy()
    out = out.drop(columns="_sparse_source_pos")
    return out, augmented


def _point_median_distances(
    points: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    road: AugmentedNetwork,
) -> gpd.GeoDataFrame:
    center_node = {
        (str(row["type"]), int(row["cluster"])): int(row["center_network_node"])
        for _, row in centers.iterrows()
    }
    keys = [
        (str(t), int(c))
        for t, c in zip(points["type"], points["cluster"], strict=True)
    ]
    missing = sorted(set(keys) - set(center_node))
    if missing:
        raise RuntimeError(f"points have no network 1-median: {missing[:10]}")
    targets = np.asarray([center_node[key] for key in keys], dtype=np.int64)
    solver = SparseRoadSolver.from_augmented(road)
    distance = solver.point_distances_from_centers(
        points["network_node"].to_numpy(np.int64, copy=False),
        targets,
    )
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise RuntimeError("one or more retained points cannot reach their own cluster median")
    out = points.copy()
    out["distance_to_cluster_median"] = distance.astype(float)
    out["point_center_distance_method"] = "network_shortest_path"
    return out


def _clean_center_output(centers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required = ["type", "cluster", "center_network_node", "geometry"]
    missing = set(required) - set(centers.columns)
    if missing:
        raise RuntimeError(f"center output is missing columns: {sorted(missing)}")
    if centers.crs is None:
        raise RuntimeError("center output is missing its CRS")
    out = _normalize_type_column(centers[required])
    out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(np.int64)
    if (out["cluster"] < 0).any():
        raise RuntimeError("center output contains a negative/noise cluster identifier")
    for position, geometry in enumerate(out.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Point":
            raise RuntimeError(f"center row {position} must contain one non-empty Point")
        coordinates = np.asarray(geometry.coords, dtype=float)
        if not np.isfinite(coordinates[:, :2]).all():
            raise RuntimeError(f"center row {position} contains non-finite coordinates")
    if len(_node_keys(out)) != len(out):
        raise RuntimeError("network centers must have exactly one row per [type, cluster]")
    return out.sort_values(["type", "cluster"], kind="stable").reset_index(drop=True)


def _clean_distance_output(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "point_id" not in points.columns:
        points = points.copy()
        points["point_id"] = (
            points["canonical_id"]
            if "canonical_id" in points.columns
            else points.index.astype(str)
        )
    columns = ["point_id", "type", "cluster", "distance_to_cluster_median"]
    columns += [c for c in ["point_center_distance_method"] if c in points.columns]
    columns += [c for c in ["canonical_id", "canonical_name"] if c in points.columns]
    columns += ["geometry"]
    return points[columns].copy()


def _clean_partition_output(partitions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required = ["type", "cluster", "geometry"]
    missing = set(required) - set(partitions.columns)
    if missing:
        raise RuntimeError(f"partition output is missing columns: {sorted(missing)}")
    if partitions.crs is None:
        raise RuntimeError("partition output is missing its CRS")
    out = _normalize_type_column(partitions[required])
    out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(np.int64)
    if (out["cluster"] < 0).any():
        raise RuntimeError("partition output contains a negative/noise cluster identifier")
    for position, geom in enumerate(out.geometry):
        if geom is None or geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise RuntimeError(f"partition row {position} is not one non-empty polygonal geometry")
        if not geom.is_valid or not np.isfinite(float(geom.area)) or float(geom.area) <= 0:
            raise RuntimeError(f"partition row {position} is invalid or has non-positive area")
    if len(_node_keys(out)) != len(out):
        raise RuntimeError("partitions must have exactly one row per [type, cluster]")
    return out.sort_values(["type", "cluster"], kind="stable").reset_index(drop=True)


def _load_io_pair(
    *,
    classification: str,
    transactions_override_path: str | None,
    transactions_sheet: str | int,
    technical_coefficients_path: str | None,
    technical_coefficients_sheet: str | int,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    transactions, info = load_engine_io_table(
        classification,
        override_path=transactions_override_path,
        sheet_name=transactions_sheet,
    )
    expected = _expected_sector_count(classification)

    if technical_coefficients_path is not None:
        coefficient_path = _require_file(
            technical_coefficients_path, "technical coefficient matrix"
        )
        coefficients = read_technical_coefficients(
            coefficient_path,
            sheet_name=technical_coefficients_sheet,
            expected_sector_count=expected,
        )
        coefficients.attrs["technical_coefficient_source"] = "explicit_matrix_override"
        coefficients.attrs["technical_coefficient_source_path"] = str(coefficient_path)
        return transactions, coefficients, info

    if transactions_override_path is None:
        total_output = load_builtin_total_output(classification)
        coefficients = derive_technical_coefficients(transactions, total_output)
        coefficients.attrs["technical_coefficient_source"] = (
            "derived_from_builtin_psa_total_output"
        )
        coefficients.attrs["technical_coefficient_source_path"] = (
            f"package:{total_output.attrs.get('resource_filename')}"
        )
        coefficients.attrs["total_output_resource_sha256"] = total_output.attrs.get(
            "resource_sha256"
        )
        return transactions, coefficients, info

    transaction_path = _require_file(
        transactions_override_path, "full transaction workbook"
    )
    total_output = read_total_output_vector(
        transaction_path,
        sheet_name=transactions_sheet,
        sector_ids=transactions.columns,
    )
    coefficients = derive_technical_coefficients(transactions, total_output)
    coefficients.attrs["technical_coefficient_source"] = "derived_from_transaction_total_output"
    coefficients.attrs["technical_coefficient_source_path"] = str(transaction_path)
    coefficients.attrs["total_output_source_column_zero_based"] = total_output.attrs.get(
        "source_column_zero_based"
    )
    return transactions, coefficients, info


def _center_distance_network(
    roads_path: str,
    roads_layer: str | None,
    centers: gpd.GeoDataFrame,
    *,
    vertex_digits: int,
    progress: bool,
) -> tuple[gpd.GeoDataFrame, AugmentedNetwork, float]:
    """Rebuild exact center-to-center road distances from saved center geometries.

    Saved augmented node IDs are intentionally ignored.  The median geometries are
    re-snapped to the supplied road linework and inserted into a fresh graph.
    """
    base = RoadNetwork.from_file(roads_path, roads_layer, vertex_digits=vertex_digits)
    projected = centers.to_crs(base.crs) if centers.crs != base.crs else centers.copy()
    snaps = base.snap(projected, progress=(lambda m: _report(True, m)) if progress else None)
    augmented = base.augment(snaps, progress=(lambda m: _report(True, m)) if progress else None)
    projected = projected.reset_index(drop=True)
    projected["center_network_node"] = augmented.point_node
    max_resnap = float(snaps.frame["snap_distance"].max()) if len(snaps.frame) else 0.0
    return projected, augmented, max_resnap


def _x_artifacts(
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
    io_dag: nx.DiGraph,
    road: AugmentedNetwork,
    *,
    centrality_direction: str,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, object]]:
    graph, disconnected = instantiate_network(
        centers,
        partitions,
        io_dag,
        road,
        progress=(lambda m: _report(True, m)) if progress else None,
    )
    centrality = directed_eigenvector_centrality(graph, centrality_direction)

    node_rows: list[dict[str, object]] = []
    for node, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        node_rows.append(
            {
                "type": str(data["type"]),
                "cluster": int(data["cluster"]),
                "centrality": float(centrality.values[str(node)]),
            }
        )

    edge_rows: list[dict[str, object]] = []
    for source, target, data in sorted(
        graph.edges(data=True), key=lambda edge: (str(edge[0]), str(edge[1]))
    ):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        edge_rows.append(
            {
                "source_type": str(source_data["type"]),
                "source_cluster": int(source_data["cluster"]),
                "target_type": str(target_data["type"]),
                "target_cluster": int(target_data["cluster"]),
                "technical_coefficient": float(data["dag_weight"]),
                "road_distance": float(data["road_distance"]),
                "edge_weight": float(data["weight"]),
            }
        )

    disconnected_rows: list[dict[str, object]] = []
    for row in disconnected.to_dict("records"):
        disconnected_rows.append(
            {
                "source_type": str(row["type1"]),
                "source_cluster": int(row["cluster1"]),
                "target_type": str(row["type2"]),
                "target_cluster": int(row["cluster2"]),
                "reason": str(row["status"]),
            }
        )

    nodes = pd.DataFrame(node_rows, columns=["type", "cluster", "centrality"])
    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "source_type",
            "source_cluster",
            "target_type",
            "target_cluster",
            "technical_coefficient",
            "road_distance",
            "edge_weight",
        ],
    )
    disconnected_out = pd.DataFrame(
        disconnected_rows,
        columns=[
            "source_type",
            "source_cluster",
            "target_type",
            "target_cluster",
            "reason",
        ],
    )
    meta = {
        "x_nodes": int(graph.number_of_nodes()),
        "x_edges": int(graph.number_of_edges()),
        "disconnected_overlap_pairs": int(len(disconnected_out)),
        "centrality_method": "directed_weighted_eigenvector",
        "centrality_direction": centrality.direction,
        "centrality_degenerate_dag": bool(centrality.degenerate_dag),
        "centrality_note": centrality.note,
    }
    return nodes, edges, disconnected_out, centrality.values, meta


def _write_late_stage_outputs(
    output_dir: Path,
    io_rows: pd.DataFrame,
    x_nodes: pd.DataFrame,
    x_edges: pd.DataFrame,
    disconnected: pd.DataFrame,
    final_points: gpd.GeoDataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "io_dag": output_dir / "sigma_io_dag.csv",
        "x_nodes": output_dir / "sigma_X_nodes.csv",
        "x_edges": output_dir / "sigma_X_edges.csv",
        "disconnected_overlaps": output_dir / "sigma_disconnected_overlap_pairs.csv",
        "points": output_dir / "sigma_points_centrality.parquet",
    }
    io_rows.to_csv(paths["io_dag"], index=False)
    x_nodes.to_csv(paths["x_nodes"], index=False)
    x_edges.to_csv(paths["x_edges"], index=False)
    disconnected.to_csv(paths["disconnected_overlaps"], index=False)
    write_geoparquet(final_points, paths["points"])
    return {key: str(path) for key, path in paths.items()}


def _run_steps_5_to_7(
    *,
    points_with_distance: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
    road: AugmentedNetwork,
    transactions: pd.DataFrame,
    technical_coefficients: pd.DataFrame,
    mwas_method: str,
    centrality_direction: str,
    distance_tempering: float,
    output_dir: Path,
    progress: bool,
) -> tuple[gpd.GeoDataFrame, dict[str, str], dict[str, object]]:
    cluster_count = _assert_cluster_invariant(points_with_distance, centers, partitions)
    _report(progress, f"Step 5: MWAS ({mwas_method}) on transactions, then A reweighting")
    io_result = build_io_dag(
        transactions,
        technical_coefficients,
        method=mwas_method,
    )
    _report(
        progress,
        f"Step 5 complete: {io_result.mwas.retained_edge_count} surviving sector edges",
    )

    _report(progress, "Step 6: positive-area partition overlaps + median road distances")
    x_nodes, x_edges, disconnected, centrality, x_meta = _x_artifacts(
        centers,
        partitions,
        io_result.graph,
        road,
        centrality_direction=centrality_direction,
        progress=progress,
    )
    if len(x_nodes) != cluster_count:
        raise RuntimeError(
            f"X-node invariant failed: expected {cluster_count} nodes, got {len(x_nodes)}"
        )

    _report(progress, "Step 7: cluster-p90 bounded distance tempering")
    final_points = tempered_point_scores(
        points_with_distance,
        centrality,
        distance_tempering=distance_tempering,
    )
    output_paths = _write_late_stage_outputs(
        output_dir,
        io_result.rows,
        x_nodes,
        x_edges,
        disconnected,
        final_points,
    )

    meta = {
        "cluster_count": cluster_count,
        "mwas_method": mwas_method,
        "mwas_solver_method": io_result.mwas.method,
        "mwas_exact": bool(io_result.mwas.exact),
        "mwas_optimal": bool(io_result.mwas.optimal),
        "mwas_source_edges": int(io_result.mwas.source_edge_count),
        "mwas_retained_edges": int(io_result.mwas.retained_edge_count),
        "distance_tempering": float(distance_tempering),
        **x_meta,
    }
    return final_points, output_paths, meta


def run_engine(config: EngineConfig) -> EngineResult:
    """Execute Steps 1--7 according to the revised workflow specification."""
    _validate_common(
        classification=config.classification,
        mwas_method=config.mwas_method,
        centrality_direction=config.centrality_direction,
        distance_tempering=config.distance_tempering,
    )
    if config.min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")
    if config.cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
    if config.hdbscan_distance_mode not in {"adaptive", "fixed"}:
        raise ValueError("hdbscan_distance_mode must be 'adaptive' or 'fixed'")

    _require_file(config.points_path, "classified points")
    _require_file(config.roads_path, "road network")
    _require_file(config.boundary_path, "study boundary")
    if config.technical_coefficients_path is not None:
        _require_file(config.technical_coefficients_path, "technical coefficient matrix")
    if config.transactions_override_path is not None:
        _require_file(config.transactions_override_path, "transaction matrix override")

    transactions, coefficients, io_info = _load_io_pair(
        classification=config.classification,
        transactions_override_path=config.transactions_override_path,
        transactions_sheet=config.transactions_sheet,
        technical_coefficients_path=config.technical_coefficients_path,
        technical_coefficients_sheet=config.technical_coefficients_sheet,
    )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _report(config.progress, "Step 1: classified point validation and network positioning")
    raw_points = read_vector(config.points_path)
    points, point_info = prepare_point_input(
        raw_points,
        classification=config.classification,
        io80_column=config.io80_column,
        io16_column=config.io16_column,
    )
    roads_gdf = read_vector(config.roads_path, config.roads_layer)
    points = points.to_crs(roads_gdf.crs)
    sparse_context = prepare_sparse_context(
        roads_gdf,
        points,
        vertex_digits=config.vertex_digits,
        progress=(lambda m: _report(True, m)) if config.progress else None,
    )
    if config.max_snap_distance is not None:
        too_far = sparse_context.snaps.snap_distance > float(config.max_snap_distance)
        if bool(too_far.any()):
            raise ValueError(
                f"{int(too_far.sum())} classified points exceed max_snap_distance="
                f"{config.max_snap_distance:g}"
            )

    _validate_point_types(points, transactions)

    _report(config.progress, "Step 2: network HDBSCAN independently within each type")
    clustering_config = ClusteringConfig(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
        allow_single_cluster=config.allow_single_cluster,
        max_distance=config.hdbscan_max_distance,
        distance_mode=config.hdbscan_distance_mode,
        min_distance=config.hdbscan_min_distance,
        distance_growth=config.hdbscan_distance_growth,
        distance_steps=config.hdbscan_distance_steps,
        core_truncation_tolerance=config.hdbscan_core_truncation_tolerance,
        stability_tolerance=config.hdbscan_stability_tolerance,
        max_neighbor_pairs=config.hdbscan_max_neighbor_pairs,
    )
    clustered = cluster_by_type(
        points,
        sparse_context,
        clustering_config,
        progress=(lambda m: _report(True, m)) if config.progress else None,
        continue_on_error=False,
    )
    clustered_public = _public_clustered(clustered, sparse_context)
    clustered_path = output_dir / "sigma_clustered_points.parquet"
    write_geoparquet(clustered_public, clustered_path)
    retained = retained_points(clustered)
    if retained.empty:
        raise RuntimeError("network HDBSCAN retained no clusters")
    retained, augmented = _build_augmented_for_retained(
        retained,
        sparse_context,
        roads_gdf.crs,
        vertex_digits=config.vertex_digits,
        progress=config.progress,
    )

    # Step 3 and Step 4 both consume the Step-2 retained points.  They are intentionally
    # independent: partition construction below never reads centers.
    _report(config.progress, "Step 3: exact network 1-median for every retained cluster")
    centers = cluster_centers(
        retained,
        augmented,
        progress=(lambda m: _report(True, m)) if config.progress else None,
        continue_on_error=False,
    )
    centers = _clean_center_output(centers)
    points_with_distance = _point_median_distances(retained, centers, augmented)
    distance_public = _clean_distance_output(points_with_distance)
    centers_path = output_dir / "sigma_network_centers.parquet"
    distance_path = output_dir / "sigma_points_with_center_distance.parquet"
    write_geoparquet(centers, centers_path)
    write_geoparquet(distance_public, distance_path)

    _report(config.progress, "Step 4: network Voronoi by point, dissolved to [type, cluster]")
    boundary = read_vector(config.boundary_path, config.boundary_layer)
    partitions = all_surface_partitions(
        augmented,
        retained,
        boundary,
        config.voronoi_resolution,
        config.voronoi_max_cells,
        progress=(lambda m: _report(True, m)) if config.progress else None,
        centers=None,
        auto_refine=True,
        refine_factor=config.voronoi_refine_factor,
        max_refinements=config.voronoi_max_refinements,
        euclidean_fallback=False,
        continue_on_error=False,
    )
    partitions = _clean_partition_output(partitions)
    partitions_path = output_dir / "sigma_partitions.parquet"
    write_geoparquet(partitions, partitions_path)
    cluster_count = _assert_cluster_invariant(points_with_distance, centers, partitions)

    final_points, late_paths, late_meta = _run_steps_5_to_7(
        points_with_distance=points_with_distance,
        centers=centers,
        partitions=partitions,
        road=augmented,
        transactions=transactions,
        technical_coefficients=coefficients,
        mwas_method=config.mwas_method,
        centrality_direction=config.centrality_direction,
        distance_tempering=config.distance_tempering,
        output_dir=output_dir,
        progress=config.progress,
    )

    paths = {
        "clustered_points": str(clustered_path),
        "centers": str(centers_path),
        "points_with_center_distance": str(distance_path),
        "partitions": str(partitions_path),
        **late_paths,
    }
    metadata: dict[str, object] = {
        "sigma_engine_version": __version__,
        "workflow_version": "revised-v0.1.0",
        "entrypoint": "full",
        "classification": config.classification,
        "classification_column": point_info.classification_column,
        "transaction_source": getattr(io_info, "source_id", None),
        "technical_coefficient_source": coefficients.attrs.get(
            "technical_coefficient_source"
        ),
        "technical_coefficient_source_path": coefficients.attrs.get(
            "technical_coefficient_source_path"
        ),
        "total_output_source_column_zero_based": coefficients.attrs.get(
            "total_output_source_column_zero_based"
        ),
        "total_output_resource_sha256": coefficients.attrs.get(
            "total_output_resource_sha256"
        ),
        "cluster_count": cluster_count,
        "stage_artifacts_are_restart_boundaries": True,
        **late_meta,
    }
    metadata_path = output_dir / "sigma_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return EngineResult(final_points, centers, partitions, paths, metadata)


def _resolve_restart_paths(config: ContinueConfig) -> tuple[Path, Path, Path]:
    partitions = _require_file(config.partitions_path, "partition artifact")
    base = partitions.parent
    centers = _require_file(
        config.centers_path or base / "sigma_network_centers.parquet",
        "network-center artifact",
    )
    points = _require_file(
        config.points_with_center_distance_path
        or base / "sigma_points_with_center_distance.parquet",
        "point-center-distance artifact",
    )
    return partitions, centers, points


def _normalize_restart_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate a saved point-distance artifact before reusing it in Step 7."""
    if points.crs is None:
        raise ValueError("point-center-distance artifact is missing its CRS")
    out = _normalize_type_column(points)
    used_legacy_distance_alias = False
    if "distance_to_cluster_median" not in out.columns:
        # Compatibility with sigma-engine 0.2.x artifacts only.  The revised public name
        # is always written on output.
        if "network_distance_to_center" in out.columns:
            used_legacy_distance_alias = True
            out["distance_to_cluster_median"] = pd.to_numeric(
                out["network_distance_to_center"], errors="coerce"
            )
        else:
            raise ValueError(
                "point-center-distance artifact needs distance_to_cluster_median "
                "(or legacy network_distance_to_center)"
            )
    required = {"type", "cluster", "distance_to_cluster_median", "geometry"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"point-center-distance artifact is missing: {sorted(missing)}")

    out["cluster"] = pd.to_numeric(out["cluster"], errors="raise").astype(np.int64)
    if (out["cluster"] < 0).any():
        raise ValueError("point-center-distance artifact contains HDBSCAN noise rows")
    distance = pd.to_numeric(out["distance_to_cluster_median"], errors="coerce").to_numpy(float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("saved point-to-median distances must be finite and non-negative")
    out["distance_to_cluster_median"] = distance

    for position, geometry in enumerate(out.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Point":
            raise ValueError(
                f"point-center-distance row {position} must contain one non-empty Point"
            )

    id_column = "point_id" if "point_id" in out.columns else "canonical_id"
    if id_column not in out.columns:
        raise ValueError("point-center-distance artifact needs point_id or canonical_id")
    ids = out[id_column].astype(str)
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{id_column} must be non-blank and unique")

    if used_legacy_distance_alias:
        if "point_center_distance_method" not in out.columns:
            raise ValueError(
                "legacy point-distance input does not identify the distance method; "
                "cannot verify that Step 7 is using road-network distance"
            )
        methods = set(out["point_center_distance_method"].dropna().astype(str))
        if methods != {"network_shortest_path"}:
            raise ValueError(
                "legacy point-distance input contains a non-network distance method: "
                f"{sorted(methods)}"
            )
    return out.reset_index(drop=True)


def _align_legacy_restart_points(
    points: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int]:
    """Align only legacy extra point rows created by 0.2.x fail-soft Voronoi."""
    center_keys = _node_keys(centers)
    partition_keys = _node_keys(partitions)
    if center_keys != partition_keys:
        raise RuntimeError(
            "restart requires exact center/partition cluster agreement; "
            f"missing partitions={sorted(center_keys-partition_keys)[:10]}, "
            f"extra partitions={sorted(partition_keys-center_keys)[:10]}"
        )
    point_keys = _node_keys(points)
    missing_point_clusters = sorted(center_keys - point_keys)
    if missing_point_clusters:
        raise RuntimeError(
            "restart point-distance artifact is missing required clusters: "
            f"{missing_point_clusters[:10]}"
        )
    extra = point_keys - center_keys
    if not extra:
        return points.reset_index(drop=True), 0
    keep = np.asarray(
        [
            (str(t), int(c)) in center_keys
            for t, c in zip(points["type"], points["cluster"], strict=True)
        ],
        dtype=bool,
    )
    dropped = int((~keep).sum())
    return points.loc[keep].reset_index(drop=True), dropped


def run_from_partitions(config: ContinueConfig) -> EngineResult:
    """Execute Steps 5--7 from already-computed Step-3/4 artifacts."""
    _validate_common(
        classification=config.classification,
        mwas_method=config.mwas_method,
        centrality_direction=config.centrality_direction,
        distance_tempering=config.distance_tempering,
    )
    _require_file(config.roads_path, "road network")
    if config.technical_coefficients_path is not None:
        _require_file(config.technical_coefficients_path, "technical coefficient matrix")
    if config.transactions_override_path is not None:
        _require_file(config.transactions_override_path, "transaction matrix override")

    transactions, coefficients, io_info = _load_io_pair(
        classification=config.classification,
        transactions_override_path=config.transactions_override_path,
        transactions_sheet=config.transactions_sheet,
        technical_coefficients_path=config.technical_coefficients_path,
        technical_coefficients_sheet=config.technical_coefficients_sheet,
    )
    partitions_path, centers_path, points_path = _resolve_restart_paths(config)

    partitions = _clean_partition_output(gpd.read_parquet(partitions_path))
    centers = _clean_center_output(gpd.read_parquet(centers_path))
    points = _normalize_restart_points(gpd.read_parquet(points_path))
    points, legacy_extra_point_rows_dropped = _align_legacy_restart_points(
        points, centers, partitions
    )
    _assert_cluster_invariant(points, centers, partitions)

    _validate_point_types(points, transactions)

    _report(config.progress, "Restart: rebuilding road graph from saved median geometries")
    centers, road, max_resnap = _center_distance_network(
        config.roads_path,
        config.roads_layer,
        centers,
        vertex_digits=config.vertex_digits,
        progress=config.progress,
    )
    partitions = partitions.to_crs(road.crs) if partitions.crs != road.crs else partitions
    points = points.to_crs(road.crs) if points.crs != road.crs else points

    output_dir = Path(config.output_dir).expanduser().resolve()
    final_points, paths, late_meta = _run_steps_5_to_7(
        points_with_distance=points,
        centers=centers,
        partitions=partitions,
        road=road,
        transactions=transactions,
        technical_coefficients=coefficients,
        mwas_method=config.mwas_method,
        centrality_direction=config.centrality_direction,
        distance_tempering=config.distance_tempering,
        output_dir=output_dir,
        progress=config.progress,
    )
    metadata: dict[str, object] = {
        "sigma_engine_version": __version__,
        "workflow_version": "revised-v0.1.0",
        "entrypoint": "from-partitions",
        "classification": config.classification,
        "transaction_source": getattr(io_info, "source_id", None),
        "technical_coefficient_source": coefficients.attrs.get(
            "technical_coefficient_source"
        ),
        "technical_coefficient_source_path": coefficients.attrs.get(
            "technical_coefficient_source_path"
        ),
        "total_output_source_column_zero_based": coefficients.attrs.get(
            "total_output_source_column_zero_based"
        ),
        "total_output_resource_sha256": coefficients.attrs.get(
            "total_output_resource_sha256"
        ),
        "input_partitions": str(partitions_path),
        "input_centers": str(centers_path),
        "input_points_with_center_distance": str(points_path),
        "maximum_center_resnap_distance": max_resnap,
        "legacy_extra_point_rows_dropped": legacy_extra_point_rows_dropped,
        **late_meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "sigma_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    paths["metadata"] = str(metadata_path)
    return EngineResult(final_points, centers, partitions, paths, metadata)


def _restore_network_x(
    nodes_path: str | Path,
    edges_path: str | Path,
    disconnected_path: str | Path,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Compatibility helper for existing 0.2.x tests/checkpoints.

    Revised v0.1.0 does not use a hidden X checkpoint; the public CSVs are the restart/audit
    artifacts.  This reader accepts either the former or revised column names.
    """
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    disconnected = pd.read_csv(disconnected_path)
    graph = nx.DiGraph()
    has_node_id = "node_id" in nodes.columns
    for _, row in nodes.iterrows():
        type_value = _normalize_type_value(row["type"])
        cluster = int(row["cluster"])
        node = (
            str(row["node_id"])
            if has_node_id and pd.notna(row["node_id"])
            else make_node_id(type_value, cluster)
        )
        graph.add_node(
            node,
            type=type_value,
            cluster=cluster,
            center_network_node=int(row.get("center_network_node", -1)),
            center_x=float(row.get("center_x", np.nan)),
            center_y=float(row.get("center_y", np.nan)),
        )
    old_schema = {"node_from", "node_to"}.issubset(edges.columns)
    for _, row in edges.iterrows():
        if old_schema:
            source, target = str(row["node_from"]), str(row["node_to"])
            dag_weight = float(row["dag_weight"])
            road_distance = float(row["road_distance"])
            weight = float(row["weight"])
        else:
            source_type = _normalize_type_value(row["source_type"])
            target_type = _normalize_type_value(row["target_type"])
            source = make_node_id(source_type, int(row["source_cluster"]))
            target = make_node_id(target_type, int(row["target_cluster"]))
            dag_weight = float(row["technical_coefficient"])
            road_distance = float(row["road_distance"])
            weight = float(row["edge_weight"])
        graph.add_edge(
            source,
            target,
            dag_weight=dag_weight,
            road_distance=road_distance,
            weight=weight,
        )
    return graph, disconnected
