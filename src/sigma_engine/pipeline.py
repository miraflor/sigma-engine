"""End-to-end orchestration for the focused SIGMA spatial-economic workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from .center import cluster_centers
from ._version import __version__
from .clustering import (
    ClusteringConfig,
    cluster_by_type,
    prepare_sparse_context,
    retained_points,
)
from .builtin_io import load_engine_io_table
from .io_dag import io_network, solve_mwas
from .io_utils import read_vector, write_geoparquet
from .network import AugmentedNetwork, RoadNetwork, SnappedPoints
from .point_input import prepare_point_input
from .progress import ProgressCallback, ProgressReporter
from .spatial_graph import directed_eigenvector_centrality, instantiate_network, make_node_id
from ._sparse_solver import SparseRoadSolver
from .voronoi import all_surface_partitions


@dataclass(frozen=True)
class EngineConfig:
    """Complete, serializable configuration for one SIGMA engine run."""

    roads_path: str
    boundary_path: str
    output_dir: str

    # Point acquisition/classification is deliberately outside SIGMA Engine.  The engine
    # consumes one already-produced classified vector file.
    points_path: str

    classification: Literal["io80", "io16"] = "io80"
    # Normal runs use the bundled PSA 2018 transaction matrix.  This path is an explicit
    # custom/experimental override only.
    io_table_override_path: str | None = None
    # None means auto-detect a recognized producer convention; a custom field can be named
    # explicitly without changing the engine.
    io80_column: str | None = None
    io16_column: str | None = None

    roads_layer: str | None = None
    boundary_layer: str | None = None
    io_sheet: str | int = 0
    mwas_method: Literal["fast", "exact"] = "fast"

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

    voronoi_resolution: float = 500.0
    voronoi_max_cells: int = 1_000_000
    voronoi_refine_factor: float = 0.5
    voronoi_max_refinements: int = 5
    fail_fast: bool = False
    vertex_digits: int = 11
    max_snap_distance: float | None = None

    centrality_direction: Literal["incoming", "outgoing"] = "incoming"
    zero_distance_floor: float = 1.0
    # CLI runs enable live progress; direct Python use remains quiet by default.
    progress: bool = False
    # Durable stage checkpoints are written on every run. Resume controls reuse.
    resume: bool = True
    restart: bool = False


@dataclass(frozen=True)
class EngineResult:
    """Primary in-memory artifacts plus paths/metadata from a completed run."""

    points: gpd.GeoDataFrame
    centers: gpd.GeoDataFrame
    partitions: gpd.GeoDataFrame
    output_paths: dict[str, str]
    metadata: dict[str, object]


def _require_existing_file(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def _validate_config(config: EngineConfig) -> None:
    """Validate cross-field and scalar configuration before expensive work starts."""
    if config.classification not in {"io80", "io16"}:
        raise ValueError("classification must be 'io80' or 'io16'")
    if config.centrality_direction not in {"incoming", "outgoing"}:
        raise ValueError("centrality_direction must be 'incoming' or 'outgoing'")
    if config.mwas_method not in {"fast", "exact"}:
        raise ValueError("mwas_method must be 'fast' or 'exact'")
    if config.cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")

    if not str(config.points_path).strip():
        raise ValueError("points_path must not be blank")
    for label, column in (("io80_column", config.io80_column), ("io16_column", config.io16_column)):
        if column is not None and not column.strip():
            raise ValueError(f"{label} must be None or a non-blank column name")

    if isinstance(config.min_cluster_size, bool) or config.min_cluster_size < 2:
        raise ValueError("min_cluster_size must be an integer >= 2")
    if config.min_samples is not None and (
        isinstance(config.min_samples, bool) or config.min_samples < 1
    ):
        raise ValueError("min_samples must be None or an integer >= 1")
    if not np.isfinite(config.hdbscan_max_distance) or config.hdbscan_max_distance <= 0:
        raise ValueError("hdbscan_max_distance must be finite and positive")
    if config.hdbscan_distance_mode not in {"adaptive", "fixed"}:
        raise ValueError("hdbscan_distance_mode must be 'adaptive' or 'fixed'")
    if config.hdbscan_min_distance is not None and (
        not np.isfinite(config.hdbscan_min_distance)
        or config.hdbscan_min_distance <= 0
        or config.hdbscan_min_distance > config.hdbscan_max_distance
    ):
        raise ValueError("hdbscan_min_distance must be None or in (0, hdbscan_max_distance]")
    if not np.isfinite(config.hdbscan_distance_growth) or config.hdbscan_distance_growth <= 1:
        raise ValueError("hdbscan_distance_growth must be finite and > 1")
    if (
        isinstance(config.hdbscan_distance_steps, bool)
        or not isinstance(config.hdbscan_distance_steps, int)
        or config.hdbscan_distance_steps < 2
    ):
        raise ValueError("hdbscan_distance_steps must be an integer >= 2")
    if not np.isfinite(config.hdbscan_core_truncation_tolerance) or not (
        0 <= config.hdbscan_core_truncation_tolerance <= 1
    ):
        raise ValueError("hdbscan_core_truncation_tolerance must be between 0 and 1")
    if not np.isfinite(config.hdbscan_stability_tolerance) or config.hdbscan_stability_tolerance < 0:
        raise ValueError("hdbscan_stability_tolerance must be finite and >= 0")
    if (
        isinstance(config.hdbscan_max_neighbor_pairs, bool)
        or config.hdbscan_max_neighbor_pairs < 1
    ):
        raise ValueError("hdbscan_max_neighbor_pairs must be a positive integer")

    if not np.isfinite(config.voronoi_resolution) or config.voronoi_resolution <= 0:
        raise ValueError("voronoi_resolution must be finite and positive")
    if (
        isinstance(config.voronoi_max_cells, bool)
        or not isinstance(config.voronoi_max_cells, int)
        or config.voronoi_max_cells < 1
    ):
        raise ValueError("voronoi_max_cells must be a positive integer")
    if not np.isfinite(config.voronoi_refine_factor) or not (0 < config.voronoi_refine_factor < 1):
        raise ValueError("voronoi_refine_factor must be finite and between 0 and 1")
    if (
        isinstance(config.voronoi_max_refinements, bool)
        or not isinstance(config.voronoi_max_refinements, int)
        or config.voronoi_max_refinements < 0
    ):
        raise ValueError("voronoi_max_refinements must be an integer >= 0")
    if not isinstance(config.fail_fast, bool):
        raise ValueError("fail_fast must be True or False")
    if not isinstance(config.resume, bool) or not isinstance(config.restart, bool):
        raise ValueError("resume and restart must be True or False")
    if isinstance(config.vertex_digits, bool) or not isinstance(config.vertex_digits, int):
        raise ValueError("vertex_digits must be an integer")
    if not 1 <= config.vertex_digits <= 15:
        raise ValueError("vertex_digits must be between 1 and 15 significant digits")

    if config.max_snap_distance is not None:
        if not np.isfinite(config.max_snap_distance) or config.max_snap_distance < 0:
            raise ValueError("max_snap_distance must be None or finite and non-negative")
    if not np.isfinite(config.zero_distance_floor) or config.zero_distance_floor <= 0:
        raise ValueError("zero_distance_floor must be finite and positive")

    _require_existing_file(config.roads_path, "road input")
    _require_existing_file(config.boundary_path, "boundary input")
    if config.io_table_override_path is not None:
        _require_existing_file(config.io_table_override_path, "IO table override")
    _require_existing_file(config.points_path, "point input")


def _validate_point_types_against_io(points: gpd.GeoDataFrame, io_table: pd.DataFrame) -> None:
    """Ensure the selected spatial classification and IO table are the same resolution."""
    point_types = set(points["type"].astype(str))
    io_types = set(str(value) for value in io_table.index)
    missing = sorted(point_types - io_types)
    if missing:
        raise ValueError(
            "classified point sectors are absent from the IO table: "
            f"{missing[:20]}. Check --classification and the selected IO table."
        )


def _attach_snap_qa(points: gpd.GeoDataFrame, snapped_frame: pd.DataFrame) -> gpd.GeoDataFrame:
    """Attach point-to-road QA fields while preserving original point geometry."""
    if len(points) != len(snapped_frame):
        raise RuntimeError("snap QA rows are misaligned with point rows")
    out = points.copy()
    out["snap_distance_to_network"] = snapped_frame["snap_distance"].to_numpy(float)
    out["snapped_edge_id"] = snapped_frame["edge_id"].to_numpy(np.int64)
    return out


def _attach_center_distances(
    points: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    road,
    progress: ProgressCallback | None = None,
    *,
    continue_on_error: bool = False,
    events: list[dict[str, object]] | None = None,
) -> gpd.GeoDataFrame:
    """Attach point-to-center distances with a Euclidean per-cluster fallback."""
    center_node = {
        (str(row["type"]), int(row["cluster"])): int(row["center_network_node"])
        for _, row in centers.iterrows()
    }
    center_geometry = {
        (str(row["type"]), int(row["cluster"])): row.geometry
        for _, row in centers.iterrows()
    }
    center_for_point = np.asarray(
        [center_node[(str(t), int(c))] for t, c in zip(points["type"], points["cluster"], strict=True)],
        dtype=np.int64,
    )
    if progress is not None:
        progress(
            f"point-center distances: SciPy shortest paths from {len(np.unique(center_for_point)):,} centers"
        )
    solver = SparseRoadSolver.from_augmented(road)
    primary_error: Exception | None = None
    try:
        distances = solver.point_distances_from_centers(
            points["network_node"].to_numpy(np.int64, copy=False), center_for_point
        )
        if not np.isfinite(distances).all():
            raise RuntimeError("one or more points cannot reach their cluster center")
    except (ValueError, RuntimeError, MemoryError) as exc:
        if not continue_on_error:
            raise
        primary_error = exc
        distances = np.full(len(points), np.nan, dtype=float)
        if progress is not None:
            progress(
                f"point-center distances: WARNING primary network calculation failed: {exc}; "
                "isolating clusters and using Euclidean fallback where necessary"
            )

    out = points.copy()
    out["node_id"] = [
        make_node_id(str(t), int(c)) for t, c in zip(out["type"], out["cluster"], strict=True)
    ]

    if primary_error is None:
        out["network_distance_to_center"] = distances
        out["point_center_distance_method"] = "network_shortest_path"
        if progress is not None:
            progress(f"point-center distances: complete for {len(out):,} points")
        return out

    out["network_distance_to_center"] = np.nan
    out["point_center_distance_method"] = "network_shortest_path"
    keep = np.ones(len(out), dtype=bool)
    for (type_value, cluster), group in out.groupby(["type", "cluster"], sort=True):
        key = (str(type_value), int(cluster))
        idx = group.index.to_numpy(np.int64)
        try:
            group_distances = solver.point_distances_from_centers(
                group["network_node"].to_numpy(np.int64, copy=False),
                np.full(len(group), center_node[key], dtype=np.int64),
            )
            if not np.isfinite(group_distances).all():
                raise RuntimeError("cluster center is unreachable by one or more member points")
            out.loc[idx, "network_distance_to_center"] = group_distances
        except (ValueError, RuntimeError, MemoryError) as exc:
            try:
                euclidean = np.asarray(
                    shapely.distance(group.geometry.to_numpy(), center_geometry[key]), dtype=float
                )
                if euclidean.shape != (len(group),) or not np.isfinite(euclidean).all():
                    raise RuntimeError("Euclidean point-center fallback produced invalid distances")
                out.loc[idx, "network_distance_to_center"] = euclidean
                out.loc[idx, "point_center_distance_method"] = "euclidean_fallback"
                if events is not None:
                    events.append(
                        {
                            "stage": "point_center_distance",
                            "type": key[0],
                            "cluster": key[1],
                            "status": "recovered",
                            "action": "fallback_euclidean_point_center_distance",
                            "attempt": 2,
                            "requested_resolution": None,
                            "used_resolution": None,
                            "error": str(exc),
                        }
                    )
                if progress is not None:
                    progress(
                        f"point-center distances: [{key[0]}, {key[1]}] recovered with "
                        "Euclidean point-center distances"
                    )
            except (ValueError, RuntimeError, MemoryError) as fallback_exc:
                keep[idx] = False
                if events is not None:
                    events.append(
                        {
                            "stage": "point_center_distance",
                            "type": key[0],
                            "cluster": key[1],
                            "status": "skipped",
                            "action": "skip_cluster_after_point_center_distance_failure",
                            "attempt": 2,
                            "requested_resolution": None,
                            "used_resolution": None,
                            "error": f"network={exc}; euclidean={fallback_exc}",
                        }
                    )
                if progress is not None:
                    progress(
                        f"point-center distances: WARNING [{key[0]}, {key[1]}] network and "
                        f"Euclidean distance failed; dropping cluster: {fallback_exc}"
                    )

    out = out.loc[keep].reset_index(drop=True)
    if progress is not None:
        progress(f"point-center distances: complete for {len(out):,} surviving points")
    return out


def _attach_node_ids(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add the same explicit [type, cluster] join key used by network X."""
    out = frame.copy()
    out["node_id"] = [
        make_node_id(str(type_value), int(cluster))
        for type_value, cluster in zip(out["type"], out["cluster"], strict=True)
    ]
    return out


def _align_to_successful_partitions(
    retained: gpd.GeoDataFrame,
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
    *,
    fail_fast: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str]]:
    """Drop cluster nodes whose type failed Voronoi rendering in fail-soft mode."""
    center_node_ids = set(centers["node_id"])
    partition_keys = set(partitions["node_id"])
    extra_partitions = sorted(partition_keys - center_node_ids)
    if extra_partitions:
        raise RuntimeError(
            "partition rows have no matching center nodes: "
            f"{extra_partitions[:10]}"
        )
    missing_partitions = sorted(center_node_ids - partition_keys)
    if missing_partitions and fail_fast:
        raise RuntimeError(
            "center/partition node keys diverged: "
            f"missing partitions={missing_partitions[:10]}"
        )
    if not missing_partitions:
        return retained, centers, []
    centers = centers.loc[centers["node_id"].isin(partition_keys)].reset_index(drop=True)
    retained = retained.loc[retained["node_id"].isin(partition_keys)].reset_index(drop=True)
    return retained, centers, missing_partitions


def _write_csv(rows: list[dict[str, object]], columns: list[str], path: Path) -> Path:
    """Write deterministic CSV headers even when an audit table has zero rows."""
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
    return path


def _restore_network_x(
    node_path: Path, edge_path: Path, disconnected_path: Path
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Restore a completed network-X stage from its durable tabular artifacts."""
    node_frame = pd.read_csv(
        node_path,
        dtype={"node_id": str, "type": str},
    )
    edge_frame = pd.read_csv(
        edge_path,
        dtype={"node_from": str, "node_to": str},
    )
    graph = nx.DiGraph()
    for row in node_frame.itertuples(index=False):
        graph.add_node(
            str(row.node_id),
            type=str(row.type),
            cluster=int(row.cluster),
            center_network_node=int(row.center_network_node),
            center_x=float(row.center_x),
            center_y=float(row.center_y),
        )
    for row in edge_frame.itertuples(index=False):
        graph.add_edge(
            str(row.node_from),
            str(row.node_to),
            dag_weight=float(row.dag_weight),
            road_distance=float(row.road_distance),
            weight=float(row.weight),
        )
    disconnected = pd.read_csv(
        disconnected_path,
        dtype={"type1": str, "type2": str},
    )
    return graph, disconnected


def _crs_linear_unit_name(crs: object) -> str | None:
    """Best-effort human-readable unit name for run metadata."""
    axis_info = getattr(crs, "axis_info", None)
    if not axis_info:
        return None
    unit_names = {getattr(axis, "unit_name", None) for axis in axis_info}
    unit_names.discard(None)
    return next(iter(unit_names)) if len(unit_names) == 1 else None



_CHECKPOINT_SCHEMA = 1
_CHECKPOINT_DIRNAME = ".sigma_checkpoints"
_STAGE_ORDER = (
    "clustering",
    "augmented_roads",
    "centers",
    "point_center_distances",
    "voronoi",
    "network_x",
)
_STAGE_EVENT_COLUMNS = [
    "stage", "type", "cluster", "status", "action", "attempt",
    "requested_resolution", "used_resolution", "error",
]


def _input_identity(path_value: str | None) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _run_signature(config: EngineConfig) -> str:
    """Stable signature for checkpoints; deliberately excludes presentation-only controls."""
    cfg = asdict(config)
    for key in ("output_dir", "progress", "resume", "restart"):
        cfg.pop(key, None)
    payload = {
        "checkpoint_schema": _CHECKPOINT_SCHEMA,
        "engine_version": __version__,
        "config": cfg,
        "inputs": {
            "points": _input_identity(config.points_path),
            "roads": _input_identity(config.roads_path),
            "boundary": _input_identity(config.boundary_path),
            "io_override": _input_identity(config.io_table_override_path),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _clean_sigma_outputs(output_dir: Path) -> None:
    checkpoint_dir = output_dir / _CHECKPOINT_DIRNAME
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    for path in output_dir.glob("sigma_*"):
        if path.is_file():
            path.unlink()


def _init_checkpoint_state(config: EngineConfig, output_dir: Path, report) -> tuple[Path, Path, dict[str, object], bool]:
    signature = _run_signature(config)
    checkpoint_dir = output_dir / _CHECKPOINT_DIRNAME
    manifest_path = checkpoint_dir / "manifest.json"
    existing = _read_json(manifest_path, {})
    valid = bool(
        existing.get("checkpoint_schema") == _CHECKPOINT_SCHEMA
        and existing.get("engine_version") == __version__
        and existing.get("run_signature") == signature
    )
    if config.restart:
        if checkpoint_dir.exists() or any(output_dir.glob("sigma_*")):
            report("checkpoint: --restart requested; removing prior SIGMA stage artifacts")
        _clean_sigma_outputs(output_dir)
        valid = False
        existing = {}
    elif existing and valid and not config.resume:
        report("checkpoint: --no-resume requested; recomputing all stages from scratch")
        _clean_sigma_outputs(output_dir)
        valid = False
        existing = {}
    elif existing and not valid:
        report("checkpoint: existing state does not match this engine/config/input set; starting fresh")
        _clean_sigma_outputs(output_dir)
        existing = {}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = existing if valid else {
        "checkpoint_schema": _CHECKPOINT_SCHEMA,
        "engine_version": __version__,
        "run_signature": signature,
        "completed_stages": [],
        "stage_events": [],
    }
    state["checkpoint_schema"] = _CHECKPOINT_SCHEMA
    state["engine_version"] = __version__
    state["run_signature"] = signature
    _atomic_json(manifest_path, state)
    return checkpoint_dir, manifest_path, state, bool(valid and config.resume)


def _checkpoint_done(state: dict[str, object], stage: str, *, allow_resume: bool) -> bool:
    return allow_resume and stage in set(state.get("completed_stages", []))


def _mark_checkpoint(
    manifest_path: Path,
    state: dict[str, object],
    stage: str,
    stage_events: list[dict[str, object]],
) -> None:
    completed = [str(x) for x in state.get("completed_stages", [])]
    completed_set = set(completed)
    completed_set.add(stage)
    ordered = [name for name in _STAGE_ORDER if name in completed_set]
    ordered.extend(
        name for name in completed if name not in _STAGE_ORDER and name != stage
    )
    if stage not in _STAGE_ORDER and stage not in ordered:
        ordered.append(stage)
    state["completed_stages"] = ordered
    state["stage_events"] = stage_events
    state["last_completed_stage"] = ordered[-1] if ordered else stage
    _atomic_json(manifest_path, state)


def _load_stage_events(state: dict[str, object]) -> list[dict[str, object]]:
    rows = state.get("stage_events", [])
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    """Atomically write a non-pickled NumPy checkpoint array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    tmp.replace(path)


def _restore_augmented_roads(
    nodes_path: Path,
    edges_path: Path,
    point_node_path: Path,
    source_pos_path: Path,
    expected_source_pos: np.ndarray,
    expected_crs: object,
) -> AugmentedNetwork:
    """Restore a portable augmented-road checkpoint and verify row alignment."""
    nodes = gpd.read_parquet(nodes_path)
    edges = gpd.read_parquet(edges_path)
    with point_node_path.open("rb") as handle:
        point_node = np.load(handle, allow_pickle=False)
    with source_pos_path.open("rb") as handle:
        source_pos = np.load(handle, allow_pickle=False)
    source_pos = np.asarray(source_pos, dtype=np.int64)
    expected_source_pos = np.asarray(expected_source_pos, dtype=np.int64)
    if source_pos.ndim != 1 or not np.array_equal(source_pos, expected_source_pos):
        raise ValueError("augmented-road checkpoint does not match retained clustering rows")
    if nodes.crs != expected_crs or edges.crs != expected_crs:
        raise ValueError("augmented-road checkpoint CRS does not match the current road input")
    augmented = AugmentedNetwork.from_checkpoint_frames(nodes, edges, point_node)
    if len(augmented.point_node) != len(expected_source_pos):
        raise ValueError("augmented-road checkpoint point count does not match retained points")
    return augmented


def _write_stage_csv(rows: list[dict[str, object]], columns: list[str], path: Path) -> Path:
    return _write_csv(rows, columns, path)


def run_engine(config: EngineConfig) -> EngineResult:
    """Execute the complete SIGMA workflow and write all final/audit artifacts."""
    report = ProgressReporter(enabled=config.progress)
    stage_events: list[dict[str, object]] = []
    report(
        f"SIGMA Engine starting: classification={config.classification}, "
        f"IO transformation=MWAS ({config.mwas_method})"
    )
    report("validating configuration and input paths")
    _validate_config(config)

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir, checkpoint_manifest_path, checkpoint_state, allow_resume = _init_checkpoint_state(
        config, output_dir, report
    )
    # A user may point 0.2.7 at an output directory left by the removed RGE
    # implementation without a reusable checkpoint manifest.  Delete only those
    # obsolete IO-stage artifacts so they cannot be mistaken for current outputs.
    for obsolete_name in (
        "sigma_io_gradient.csv",
        "sigma_io_remainder.csv",
        "sigma_io_rge_nodes.csv",
        "sigma_io_rge_trace.csv",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
            report(f"outputs: removed obsolete artifact {obsolete_name}")
    if allow_resume:
        stage_events = _load_stage_events(checkpoint_state)
        completed = checkpoint_state.get("completed_stages", [])
        if completed:
            report("checkpoint: reusable stages found: " + ", ".join(str(x) for x in completed))

    # ------------------------------------------------------------------ point input
    # Acquisition/classification is an upstream concern.  SIGMA Engine starts from one
    # already-produced classified vector file and normalizes only its narrow required schema.
    points_path = Path(config.points_path).expanduser().resolve()
    report(f"points: reading {points_path}")
    raw_points = read_vector(points_path)
    raw_point_count = len(raw_points)
    report(f"points: loaded {raw_point_count:,} rows; normalizing classification")
    points, point_input_info = prepare_point_input(
        raw_points,
        classification=config.classification,
        io80_column=config.io80_column,
        io16_column=config.io16_column,
    )
    report(
        f"points: using column {point_input_info.classification_column!r}; "
        f"{len(points):,} classified, {raw_point_count - len(points):,} dropped unclassified, "
        f"{points['type'].nunique():,} types"
    )
    if len(points):
        largest = points["type"].astype(str).value_counts().head(5)
        report(
            "points: largest types: "
            + ", ".join(f"{code}={int(count):,}" for code, count in largest.items())
        )

    # Validate classification compatibility *before* road-distance work.  This catches the
    # common mistake of pairing IO16 points with an IO80 transaction table (or vice versa).
    report("IO: loading transaction matrix")
    io_table, io_input_info = load_engine_io_table(
        config.classification,
        override_path=config.io_table_override_path,
        sheet_name=config.io_sheet,
    )
    _validate_point_types_against_io(points, io_table)
    io_graph = io_network(io_table)
    report(
        f"IO: {io_input_info.source_id}; {io_graph.number_of_nodes():,} sectors, "
        f"{io_graph.number_of_edges():,} positive transaction edges"
    )

    # ------------------------------------------------------------------ sparse road clustering
    report(f"roads: reading {config.roads_path}")
    roads_gdf = read_vector(config.roads_path, config.roads_layer)
    points = points.to_crs(roads_gdf.crs)
    classified_point_count = len(points)
    sparse_context = prepare_sparse_context(
        roads_gdf,
        points,
        vertex_digits=config.vertex_digits,
        progress=report,
    )
    max_observed_snap = float(sparse_context.snaps.snap_distance.max(initial=0.0))
    if config.max_snap_distance is not None:
        too_far = sparse_context.snaps.snap_distance > config.max_snap_distance
        if bool(too_far.any()):
            count = int(too_far.sum())
            worst = float(sparse_context.snaps.snap_distance[too_far].max())
            raise ValueError(
                f"{count:,} classified points exceed max_snap_distance="
                f"{config.max_snap_distance:g}; largest snap distance is {worst:g} "
                "in the road CRS units"
            )

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
    clustering_summary_columns = [
        "type", "clustering_method", "n_points", "n_positions", "n_pairs", "n_clusters",
        "noise_share", "share_core_structurally_unreachable", "share_core_truncated",
        "max_distance", "requested_max_distance", "distance_mode", "distance_status",
        "distance_steps_run", "distance_stable", "pair_limit_encountered",
        "pair_limit_distance", "core_truncation_tolerance",
    ]
    clustering_trace_columns = [
        "type", "trial", "radius", "outcome", "n_pairs", "n_clusters", "noise_share",
        "share_core_truncated", "same_as_previous", "seconds",
    ]
    clustering_checkpoint = checkpoint_dir / "clustered_internal.parquet"
    clustering_public = output_dir / "sigma_clustered_points.parquet"
    clustering_summary_file = output_dir / "sigma_clustering_summary.csv"
    clustering_trace_file = output_dir / "sigma_clustering_distance_trace.csv"
    if (
        _checkpoint_done(checkpoint_state, "clustering", allow_resume=allow_resume)
        and clustering_checkpoint.exists() and clustering_public.exists()
        and clustering_summary_file.exists() and clustering_trace_file.exists()
    ):
        report("checkpoint: restoring completed clustering stage")
        clustered = gpd.read_parquet(clustering_checkpoint)
        clustering_summary = pd.read_csv(clustering_summary_file).where(pd.notna, None).to_dict("records") if clustering_summary_file.exists() else []
        clustering_trace = pd.read_csv(clustering_trace_file).where(pd.notna, None).to_dict("records") if clustering_trace_file.exists() else []
    else:
        report(
            f"clustering: starting sparse network HDBSCAN*; "
            f"min_cluster_size={config.min_cluster_size}, "
            f"mode={config.hdbscan_distance_mode}, ceiling={config.hdbscan_max_distance:g}"
        )
        clustered = cluster_by_type(
            points, sparse_context, clustering_config, progress=report,
            continue_on_error=not config.fail_fast, events=stage_events,
        )
        clustering_summary = list(clustered.attrs.get("clustering_summary", []))
        clustering_trace = list(clustered.attrs.get("clustering_trace", []))
        report("checkpoint: writing clustering artifacts")
        write_geoparquet(clustered, clustering_checkpoint)
        public_clustered = clustered.drop(columns=["_sparse_source_pos"], errors="ignore")
        write_geoparquet(public_clustered, clustering_public)
        _write_stage_csv(clustering_summary, clustering_summary_columns, clustering_summary_file)
        _write_stage_csv(clustering_trace, clustering_trace_columns, clustering_trace_file)
        _mark_checkpoint(checkpoint_manifest_path, checkpoint_state, "clustering", stage_events)
        _write_stage_csv(stage_events, _STAGE_EVENT_COLUMNS, output_dir / "sigma_stage_events.csv")
    for row in clustering_summary:
        status = str(row.get("distance_status"))
        if status == "pair_cap_limited":
            report(
                f"WARNING: clustering type {row['type']!r} stopped below the requested "
                "distance because max_neighbor_pairs was reached"
            )
        elif (
            status == "ceiling_reached"
            and float(row.get("share_core_truncated") or 0.0)
            > float(row.get("core_truncation_tolerance") or 0.0)
        ):
            report(
                f"WARNING: clustering type {row['type']!r} reached the distance ceiling "
                f"with {float(row['share_core_truncated']):.3%} resolvable core truncation"
            )
    retained = retained_points(clustered)
    hdbscan_retained_count = int(len(retained))
    report(
        f"clustering: complete; retained {len(retained):,}/{classified_point_count:,} points "
        f"({classified_point_count - len(retained):,} noise or skipped type failures)"
    )
    if retained.empty:
        raise ValueError("network HDBSCAN retained no clustered points")

    # Build SIGMA's exact continuous-position augmented graph only for retained observations.
    # This preserves downstream center/distance semantics while avoiding augmentation work for
    # HDBSCAN noise that will never enter the economic network.  The augmented graph is a
    # durable checkpoint because production road insertion can be material even when every
    # later analytical stage is already resumable.
    source_pos = retained["_sparse_source_pos"].to_numpy(np.int64, copy=False)
    augmented_nodes_checkpoint = checkpoint_dir / "augmented_road_nodes.parquet"
    augmented_edges_checkpoint = checkpoint_dir / "augmented_road_edges.parquet"
    augmented_point_node_checkpoint = checkpoint_dir / "augmented_road_point_node.npy"
    augmented_source_pos_checkpoint = checkpoint_dir / "augmented_road_source_pos.npy"
    augmented_files = (
        augmented_nodes_checkpoint,
        augmented_edges_checkpoint,
        augmented_point_node_checkpoint,
        augmented_source_pos_checkpoint,
    )
    augmented = None
    if (
        _checkpoint_done(checkpoint_state, "augmented_roads", allow_resume=allow_resume)
        and all(path.exists() for path in augmented_files)
    ):
        try:
            augmented = _restore_augmented_roads(
                augmented_nodes_checkpoint,
                augmented_edges_checkpoint,
                augmented_point_node_checkpoint,
                augmented_source_pos_checkpoint,
                source_pos,
                roads_gdf.crs,
            )
            report("checkpoint: restoring completed augmented road graph")
        except Exception as exc:
            report(
                "checkpoint: augmented road graph is unreadable or incompatible; "
                f"rebuilding ({exc})"
            )
            augmented = None

    if augmented is None:
        report("roads: adapting sparse graph and inserting retained point positions")
        roads = RoadNetwork.from_sparse_graph(
            sparse_context.graph, roads_gdf.crs, vertex_digits=config.vertex_digits
        )
        sparse_snaps = sparse_context.snaps.subset(source_pos)
        snapped_retained = SnappedPoints(
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
        augmented = roads.augment(snapped_retained, progress=report)
        report("checkpoint: writing augmented road graph")
        write_geoparquet(augmented.nodes, augmented_nodes_checkpoint)
        write_geoparquet(augmented.edges, augmented_edges_checkpoint)
        _atomic_npy(
            augmented_point_node_checkpoint,
            augmented.point_node.astype(np.int64, copy=False),
        )
        _atomic_npy(augmented_source_pos_checkpoint, source_pos.astype(np.int64, copy=False))
        _mark_checkpoint(
            checkpoint_manifest_path, checkpoint_state, "augmented_roads", stage_events
        )

    retained["network_node"] = augmented.point_node
    retained = retained.drop(columns="_sparse_source_pos")
    component_by_node = augmented.component_id()
    retained["network_component"] = [
        component_by_node[int(node)] for node in augmented.point_node
    ]
    report(
        f"roads: retained augmented graph ready: {augmented.graph.number_of_nodes():,} nodes, "
        f"{augmented.graph.number_of_edges():,} edges"
    )

    # ------------------------------------------------------------------ centers/partitions
    centers_checkpoint = checkpoint_dir / "centers.parquet"
    retained_centers_checkpoint = checkpoint_dir / "retained_after_centers.parquet"
    centers_public = output_dir / "sigma_network_centers.parquet"
    if (
        _checkpoint_done(checkpoint_state, "centers", allow_resume=allow_resume)
        and centers_checkpoint.exists() and retained_centers_checkpoint.exists()
    ):
        report("checkpoint: restoring completed 1-median stage")
        centers = gpd.read_parquet(centers_checkpoint)
        retained = gpd.read_parquet(retained_centers_checkpoint)
    else:
        report("centers: solving exact network 1-medians")
        centers = _attach_node_ids(
            cluster_centers(
                retained, augmented, progress=report, continue_on_error=not config.fail_fast,
                events=stage_events,
            )
        )
        if centers.empty:
            raise RuntimeError("no cluster centers could be solved")
        center_keys = set(zip(centers["type"].astype(str), centers["cluster"].astype(int), strict=True))
        has_center = np.asarray(
            [(str(t), int(c)) in center_keys for t, c in zip(retained["type"], retained["cluster"], strict=True)],
            dtype=bool,
        )
        if not has_center.all():
            dropped = int((~has_center).sum())
            report(f"centers: WARNING dropping {dropped:,} points from clusters whose center failed")
            retained = retained.loc[has_center].reset_index(drop=True)
        report(f"centers: {len(centers):,} centers solved")
        report("checkpoint: writing 1-median artifacts")
        write_geoparquet(centers, centers_checkpoint)
        write_geoparquet(retained, retained_centers_checkpoint)
        write_geoparquet(centers, centers_public)
        _mark_checkpoint(checkpoint_manifest_path, checkpoint_state, "centers", stage_events)
        _write_stage_csv(stage_events, _STAGE_EVENT_COLUMNS, output_dir / "sigma_stage_events.csv")
    distance_checkpoint = checkpoint_dir / "points_with_center_distance.parquet"
    distance_centers_checkpoint = checkpoint_dir / "centers_after_distance.parquet"
    distance_public = output_dir / "sigma_points_with_center_distance.parquet"
    if (
        _checkpoint_done(checkpoint_state, "point_center_distances", allow_resume=allow_resume)
        and distance_checkpoint.exists() and distance_centers_checkpoint.exists()
    ):
        report("checkpoint: restoring completed point-center distance stage")
        retained = gpd.read_parquet(distance_checkpoint)
        centers = gpd.read_parquet(distance_centers_checkpoint)
    else:
        report("point-center distances: computing shortest-path distances")
        retained = _attach_center_distances(
            retained, centers, augmented, progress=report, continue_on_error=not config.fail_fast,
            events=stage_events,
        )
        if retained.empty:
            raise RuntimeError("no points survived point-center distance calculation")
        surviving_center_keys = set(zip(retained["type"].astype(str), retained["cluster"].astype(int), strict=True))
        before_distance_centers = len(centers)
        centers = centers.loc[
            [(str(t), int(c)) in surviving_center_keys for t, c in zip(centers["type"], centers["cluster"], strict=True)]
        ].reset_index(drop=True)
        if len(centers) != before_distance_centers:
            report(
                f"point-center distances: WARNING dropped "
                f"{before_distance_centers - len(centers):,} centers whose clusters could not be recovered"
            )
        report("checkpoint: writing point-center-distance artifacts")
        write_geoparquet(retained, distance_checkpoint)
        write_geoparquet(centers, distance_centers_checkpoint)
        write_geoparquet(retained, distance_public)
        write_geoparquet(centers, centers_public)
        _mark_checkpoint(checkpoint_manifest_path, checkpoint_state, "point_center_distances", stage_events)
        _write_stage_csv(stage_events, _STAGE_EVENT_COLUMNS, output_dir / "sigma_stage_events.csv")

    partitions_checkpoint = checkpoint_dir / "partitions.parquet"
    partitioned_points_checkpoint = checkpoint_dir / "points_after_voronoi.parquet"
    partitioned_centers_checkpoint = checkpoint_dir / "centers_after_voronoi.parquet"
    partitions_public = output_dir / "sigma_partitions.parquet"
    partitioned_points_public = output_dir / "sigma_points_partitioned.parquet"
    if (
        _checkpoint_done(checkpoint_state, "voronoi", allow_resume=allow_resume)
        and partitions_checkpoint.exists() and partitioned_points_checkpoint.exists()
        and partitioned_centers_checkpoint.exists()
    ):
        report("checkpoint: restoring completed Voronoi stage")
        partitions = gpd.read_parquet(partitions_checkpoint)
        retained = gpd.read_parquet(partitioned_points_checkpoint)
        centers = gpd.read_parquet(partitioned_centers_checkpoint)
    else:
        report(f"boundary: reading {config.boundary_path}")
        boundary = read_vector(config.boundary_path, config.boundary_layer)
        report(
            f"voronoi: rendering partitions at resolution={config.voronoi_resolution:g} "
            f"with max_cells={config.voronoi_max_cells:,}"
        )
        partitions = _attach_node_ids(
            all_surface_partitions(
                augmented, retained, boundary, config.voronoi_resolution, config.voronoi_max_cells,
                progress=report, centers=centers, auto_refine=True,
                refine_factor=config.voronoi_refine_factor,
                max_refinements=config.voronoi_max_refinements, euclidean_fallback=True,
                continue_on_error=not config.fail_fast, events=stage_events,
            )
        )
        if partitions.empty:
            raise RuntimeError("no Voronoi type produced a usable surface partition")
        report(f"voronoi: complete; {len(partitions):,} partitions")
        retained, centers, missing_partitions = _align_to_successful_partitions(
            retained, centers, partitions, fail_fast=config.fail_fast,
        )
        if missing_partitions:
            report(
                f"voronoi: WARNING dropping {len(missing_partitions):,} cluster nodes whose type "
                "could not produce a surface"
            )
            if retained.empty or centers.empty:
                raise RuntimeError("all retained clusters were removed by recoverable stage failures")
        report("checkpoint: writing Voronoi artifacts")
        write_geoparquet(partitions, partitions_checkpoint)
        write_geoparquet(retained, partitioned_points_checkpoint)
        write_geoparquet(centers, partitioned_centers_checkpoint)
        write_geoparquet(partitions, partitions_public)
        write_geoparquet(retained, partitioned_points_public)
        write_geoparquet(centers, centers_public)
        _mark_checkpoint(checkpoint_manifest_path, checkpoint_state, "voronoi", stage_events)
        _write_stage_csv(stage_events, _STAGE_EVENT_COLUMNS, output_dir / "sigma_stage_events.csv")

    # ------------------------------------------------------ IO MWAS and X
    # The maximum-weight acyclic subgraph (MWAS) problem is the weighted
    # complement of minimum feedback arc set.  The fast default is SIGMA's
    # deterministic descending-weight cycle-avoidance heuristic; it is intentionally
    # heuristic and does not claim a global optimum.  Exact mode uses a topological-
    # rank MILP with zero relative MIP gap.  See io_dag.py for references and the
    # precise formulation.
    report(
        f"MWAS: starting {config.mwas_method} transformation on "
        f"{io_graph.number_of_nodes():,} nodes / {io_graph.number_of_edges():,} edges"
    )
    mwas = solve_mwas(io_graph, method=config.mwas_method)
    retained_fraction = (
        float(mwas.retained_weight / mwas.source_weight) if mwas.source_weight > 0 else None
    )
    optimal_note = "proven optimum" if mwas.optimal else "heuristic result"
    report(
        f"MWAS: complete; {mwas.retained_edge_count:,} retained edges; "
        f"weight={mwas.retained_weight:,.6g}; "
        f"removed={mwas.removed_weight:,.6g}; {optimal_note}"
    )

    io_dag_rows = [
        {
            "type_from": str(source),
            "type_to": str(target),
            "io_weight": float(data["weight"]),
        }
        for source, target, data in sorted(
            mwas.graph.edges(data=True), key=lambda edge: (str(edge[0]), str(edge[1]))
        )
    ]
    io_dag_path = _write_csv(
        io_dag_rows,
        ["type_from", "type_to", "io_weight"],
        output_dir / "sigma_io_dag.csv",
    )
    report(f"MWAS: artifact written: {io_dag_path.name}")

    x_nodes_path = output_dir / "sigma_X_nodes.csv"
    x_edges_path = output_dir / "sigma_X_edges.csv"
    x_disconnected_path = output_dir / "sigma_disconnected_overlap_pairs.csv"
    if (
        _checkpoint_done(checkpoint_state, "network_x", allow_resume=allow_resume)
        and x_nodes_path.exists()
        and x_edges_path.exists()
        and x_disconnected_path.exists()
    ):
        report("checkpoint: restoring completed network X stage")
        network_x, disconnected = _restore_network_x(
            x_nodes_path, x_edges_path, x_disconnected_path
        )
    else:
        report("network X: instantiating spatial overlaps and center distances")
        network_x, disconnected = instantiate_network(
            centers, partitions, mwas.graph, augmented, progress=report,
        )
        raw_node_rows = [
            {
                "node_id": str(node), "type": str(data["type"]), "cluster": int(data["cluster"]),
                "center_network_node": int(data["center_network_node"]),
                "center_x": float(data["center_x"]), "center_y": float(data["center_y"]),
                "eigenvector_centrality": None,
            }
            for node, data in sorted(network_x.nodes(data=True), key=lambda item: str(item[0]))
        ]
        _write_csv(
            raw_node_rows,
            ["node_id", "type", "cluster", "center_network_node", "center_x", "center_y", "eigenvector_centrality"],
            x_nodes_path,
        )
        raw_edge_rows = [
            {"node_from": str(source), "node_to": str(target), "dag_weight": float(data["dag_weight"]),
             "road_distance": float(data["road_distance"]), "weight": float(data["weight"])}
            for source, target, data in sorted(network_x.edges(data=True), key=lambda edge: (str(edge[0]), str(edge[1])))
        ]
        _write_csv(
            raw_edge_rows,
            ["node_from", "node_to", "dag_weight", "road_distance", "weight"],
            x_edges_path,
        )
        disconnected_rows = disconnected.to_dict("records") if not disconnected.empty else []
        _write_csv(
            disconnected_rows,
            ["type1", "cluster1", "type2", "cluster2", "status"],
            x_disconnected_path,
        )
        report(
            f"network X: complete; {network_x.number_of_nodes():,} nodes, "
            f"{network_x.number_of_edges():,} edges"
        )
        report("checkpoint: writing network X artifacts")
        _mark_checkpoint(checkpoint_manifest_path, checkpoint_state, "network_x", stage_events)
        _write_stage_csv(stage_events, _STAGE_EVENT_COLUMNS, output_dir / "sigma_stage_events.csv")

    report(
        f"centrality: computing directed {config.centrality_direction} eigenvector convention"
    )
    centrality = directed_eigenvector_centrality(
        network_x,
        config.centrality_direction,
    )

    point_node_ids = set(retained["node_id"])
    missing_centrality = sorted(point_node_ids - set(centrality.values))
    if missing_centrality:
        raise RuntimeError(
            f"centrality is missing retained SIGMA nodes: {missing_centrality[:10]}"
        )
    retained["eigenvector_centrality"] = retained["node_id"].map(centrality.values).astype(float)
    report(
        f"centrality: complete; degenerate_dag={centrality.degenerate_dag}"
    )

    # ------------------------------------------------------------------ point score
    report("scoring: computing point centrality scores")
    distance = retained["network_distance_to_center"].to_numpy(float)
    eigenvector = retained["eigenvector_centrality"].to_numpy(float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise RuntimeError("point-to-center distances are invalid")
    if not np.isfinite(eigenvector).all():
        raise RuntimeError("eigenvector centrality contains non-finite values")

    score = np.empty_like(eigenvector, dtype=float)
    at_center = distance == 0.0
    score[at_center] = eigenvector[at_center] / config.zero_distance_floor
    score[~at_center] = eigenvector[~at_center] / distance[~at_center]
    if not np.isfinite(score).all():
        raise RuntimeError("centrality scoring produced non-finite values")
    retained["centrality_score"] = score

    final_columns = [
        "canonical_id",
        "canonical_name",
        "type",
        "cluster",
        "node_id",
        "network_component",
        "snapped_edge_id",
        "snap_distance_to_network",
        "network_distance_to_center",
        "point_center_distance_method",
        "eigenvector_centrality",
        "centrality_score",
        "geometry",
    ]
    final = retained[final_columns].copy()
    # Stable output order makes audit diffs and downstream joins independent of the input
    # file's row order.  The identifier value itself is preserved; only its textual shadow
    # participates in sorting.
    final = (
        final.assign(_canonical_sort_key=final["canonical_id"].astype(str))
        .sort_values(["type", "cluster", "_canonical_sort_key"], kind="stable")
        .drop(columns="_canonical_sort_key")
        .reset_index(drop=True)
    )
    centers = centers.sort_values(["type", "cluster"], kind="stable").reset_index(drop=True)
    partitions = partitions.sort_values(["type", "cluster"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------ geospatial outputs
    report(f"outputs: writing results to {output_dir}")
    paths: dict[str, str] = {
        "points": str(write_geoparquet(final, output_dir / "sigma_points_centrality.parquet")),
        "clustered_points": str(clustering_public),
        "centers": str(write_geoparquet(centers, centers_public)),
        "points_with_center_distance": str(distance_public),
        "partitions": str(write_geoparquet(partitions, partitions_public)),
        "points_partitioned": str(partitioned_points_public),
        "checkpoint_manifest": str(checkpoint_manifest_path),
    }

    report("outputs: geospatial parquet files written; writing audit CSVs")

    # ------------------------------------------------------------------ recoverable stage events
    stage_events_path = _write_csv(
        stage_events,
        _STAGE_EVENT_COLUMNS,
        output_dir / "sigma_stage_events.csv",
    )
    paths["stage_events"] = str(stage_events_path)

    # ------------------------------------------------------------------ tabular audit outputs
    clustering_summary_path = _write_csv(
        clustering_summary,
        clustering_summary_columns,
        output_dir / "sigma_clustering_summary.csv",
    )
    paths["clustering_summary"] = str(clustering_summary_path)

    clustering_trace_path = _write_csv(
        clustering_trace,
        clustering_trace_columns,
        output_dir / "sigma_clustering_distance_trace.csv",
    )
    paths["clustering_distance_trace"] = str(clustering_trace_path)

    paths["io_dag"] = str(io_dag_path)

    node_rows = [
        {
            "node_id": str(node),
            "type": str(data["type"]),
            "cluster": int(data["cluster"]),
            "center_network_node": int(data["center_network_node"]),
            "center_x": float(data["center_x"]),
            "center_y": float(data["center_y"]),
            "eigenvector_centrality": float(centrality.values[str(node)]),
        }
        for node, data in sorted(network_x.nodes(data=True), key=lambda item: str(item[0]))
    ]
    node_path = _write_csv(
        node_rows,
        [
            "node_id",
            "type",
            "cluster",
            "center_network_node",
            "center_x",
            "center_y",
            "eigenvector_centrality",
        ],
        output_dir / "sigma_X_nodes.csv",
    )
    paths["X_nodes"] = str(node_path)

    edge_rows = [
        {
            "node_from": str(source),
            "node_to": str(target),
            "dag_weight": float(data["dag_weight"]),
            "road_distance": float(data["road_distance"]),
            "weight": float(data["weight"]),
        }
        for source, target, data in sorted(
            network_x.edges(data=True),
            key=lambda edge: (str(edge[0]), str(edge[1])),
        )
    ]
    edge_path = _write_csv(
        edge_rows,
        ["node_from", "node_to", "dag_weight", "road_distance", "weight"],
        output_dir / "sigma_X_edges.csv",
    )
    paths["X_edges"] = str(edge_path)

    if not disconnected.empty:
        disconnected = disconnected.sort_values(
            ["type1", "cluster1", "type2", "cluster2"],
            kind="stable",
        ).reset_index(drop=True)
        disconnected_path = output_dir / "sigma_disconnected_overlap_pairs.csv"
        disconnected.to_csv(disconnected_path, index=False)
        paths["disconnected_overlap_pairs"] = str(disconnected_path)

    # ------------------------------------------------------------------ metadata
    total_io_weight = float(mwas.source_weight)
    metadata: dict[str, object] = {
        "classification": config.classification,
        "points_source": "file",
        "points_path": str(points_path),
        "point_classification_column": point_input_info.classification_column,
        "canonical_name_source": point_input_info.canonical_name_source,
        "point_count_input": int(raw_point_count),
        "point_count_input_classified": int(classified_point_count),
        "point_count_dropped_unclassified": int(raw_point_count - classified_point_count),
        "point_count_retained": int(len(final)),
        "point_count_hdbscan_retained": int(hdbscan_retained_count),
        "point_count_hdbscan_noise": int(classified_point_count - hdbscan_retained_count),
        "point_count_dropped_recoverable_failures": int(hdbscan_retained_count - len(final)),
        "recoverable_stage_event_count": int(len(stage_events)),
        "recoverable_stage_skipped_count": int(sum(row.get("status") == "skipped" for row in stage_events)),
        "recoverable_stage_recovered_count": int(sum(row.get("status") == "recovered" for row in stage_events)),
        "stage_events_file": "sigma_stage_events.csv",
        "stage_events": stage_events,
        "cluster_count": int(len(centers)),
        "center_method_counts": {
            str(method): int(count)
            for method, count in centers["center_method"].astype(str).value_counts().sort_index().items()
        } if "center_method" in centers.columns else {},
        "point_center_distance_method_counts": {
            str(method): int(count)
            for method, count in final["point_center_distance_method"].astype(str).value_counts().sort_index().items()
        } if "point_center_distance_method" in final.columns else {},
        "partition_count": int(len(partitions)),
        "type_count": int(final["type"].nunique()),
        "road_crs": str(roads.crs),
        "road_linear_unit": _crs_linear_unit_name(roads.crs),
        "road_node_count_base": int(roads.graph.number_of_nodes()),
        "road_edge_count_base": int(roads.graph.number_of_edges()),
        "road_node_count_augmented": int(augmented.graph.number_of_nodes()),
        "road_edge_count_augmented": int(augmented.graph.number_of_edges()),
        "spatial_backend": "sparse-scipy-shapely",
        "hdbscan_distance_mode": config.hdbscan_distance_mode,
        "hdbscan_max_distance": config.hdbscan_max_distance,
        "hdbscan_max_neighbor_pairs": config.hdbscan_max_neighbor_pairs,
        "clustering_method_counts": {
            str(method): int(count)
            for method, count in pd.Series(
                [row.get("clustering_method", "network_hdbscan") for row in clustering_summary],
                dtype="object",
            ).value_counts().sort_index().items()
        },
        "euclidean_hdbscan_fallback_types": [
            str(row["type"])
            for row in clustering_summary
            if row.get("clustering_method") == "euclidean_hdbscan_fallback"
        ],
        "hdbscan_distance_status_counts": {
            str(status): int(count)
            for status, count in pd.Series(
                [row.get("distance_status") for row in clustering_summary], dtype="object"
            ).value_counts().sort_index().items()
        },
        "hdbscan_selected_distance_by_type": {
            str(row["type"]): float(row["max_distance"])
            for row in clustering_summary
            if row.get("clustering_method", "network_hdbscan") == "network_hdbscan"
        },
        "hdbscan_pair_cap_limited_types": [
            str(row["type"])
            for row in clustering_summary
            if row.get("distance_status") == "pair_cap_limited"
        ],
        "hdbscan_clustering_summary_file": "sigma_clustering_summary.csv",
        "hdbscan_distance_trace_file": "sigma_clustering_distance_trace.csv",
        "max_observed_snap_distance": max_observed_snap,
        "max_snap_distance": config.max_snap_distance,
        "io_sector_count": int(io_graph.number_of_nodes()),
        "io_source": io_input_info.source_id,
        "io_source_kind": io_input_info.source_kind,
        "io_source_path": io_input_info.source_path,
        "io_source_url": io_input_info.source_url,
        "io_reference_year": io_input_info.reference_year,
        "io_resource_sha256": io_input_info.resource_sha256,
        "io_table_layout": io_table.attrs.get("source_layout"),
        "io_edge_count_source": int(io_graph.number_of_edges()),
        "io_total_weight_source": total_io_weight,
        "io_dag_edge_count": int(mwas.retained_edge_count),
        "io_dag_weight": float(mwas.retained_weight),
        "io_dag_weight_fraction": retained_fraction,
        "io_removed_edge_count": int(mwas.removed_edge_count),
        "io_removed_weight": float(mwas.removed_weight),
        "io_self_loop_weight": float(mwas.self_loop_weight),
        "mwas_method": mwas.method,
        "mwas_requested_mode": config.mwas_method,
        "mwas_exact": bool(mwas.exact),
        "mwas_optimal": bool(mwas.optimal),
        "mwas_solver_status": mwas.solver_status,
        "io_dag_file": "sigma_io_dag.csv",
        "X_node_count": int(network_x.number_of_nodes()),
        "X_edge_count": int(network_x.number_of_edges()),
        "disconnected_overlap_pair_count": int(len(disconnected)),
        "centrality_direction": centrality.direction,
        "eigenvector_degenerate_dag": bool(centrality.degenerate_dag),
        "eigenvector_note": centrality.note,
        "zero_distance_policy": (
            f"EC / {config.zero_distance_floor:g} when road distance is exactly zero"
        ),
        "voronoi_surface_resolution_requested": float(config.voronoi_resolution),
        "voronoi_auto_refine": True,
        "voronoi_refine_factor": float(config.voronoi_refine_factor),
        "voronoi_max_refinements": int(config.voronoi_max_refinements),
        "voronoi_network_attempts_max": int(config.voronoi_max_refinements + 1),
        "voronoi_euclidean_fallback": True,
        "voronoi_surface_method_counts": {
            str(method): int(count)
            for method, count in partitions["surface_method"].astype(str).value_counts().sort_index().items()
        } if "surface_method" in partitions.columns else {},
        "euclidean_voronoi_fallback_types": sorted(
            partitions.loc[
                partitions.get("surface_method", pd.Series(index=partitions.index, dtype=object))
                .astype(str).eq("euclidean_voronoi_fallback"),
                "type",
            ].astype(str).unique().tolist()
        ) if "surface_method" in partitions.columns else [],
        "voronoi_surface_resolution_by_type": {
            str(type_value): (
                float(group["surface_resolution"].iloc[0])
                if np.isfinite(float(group["surface_resolution"].iloc[0])) else None
            )
            for type_value, group in partitions.groupby("type", sort=True)
        },
        "fail_fast": bool(config.fail_fast),
        "checkpoint_schema": _CHECKPOINT_SCHEMA,
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_completed_stages": list(checkpoint_state.get("completed_stages", [])),
        "resume_enabled": bool(config.resume),
        "config": asdict(config),
    }

    report("outputs: writing run metadata")
    metadata_path = output_dir / "sigma_run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path)
    checkpoint_state["final_output_complete"] = True
    checkpoint_state["stage_events"] = stage_events
    _atomic_json(checkpoint_manifest_path, checkpoint_state)

    # Repeat every recovery/skip at the end so a long run has one definitive audit summary
    # without requiring the user to scroll back through hours of progress output.
    if stage_events:
        recovered_count = sum(row.get("status") == "recovered" for row in stage_events)
        skipped_count = sum(row.get("status") == "skipped" for row in stage_events)
        report(
            f"adjustments: {len(stage_events):,} total; {recovered_count:,} recovered, "
            f"{skipped_count:,} skipped; audit={stage_events_path.name}"
        )
        for row in stage_events:
            subject = f"type={row.get('type')}"
            if row.get("cluster") is not None:
                subject += f", cluster={row.get('cluster')}"
            detail = f"; {row.get('error')}" if row.get("error") else ""
            report(
                f"adjustment: stage={row.get('stage')}, {subject}, "
                f"status={row.get('status')}, action={row.get('action')}{detail}"
            )
    else:
        report("adjustments: none; all primary methods completed without fallback/refinement")

    report(
        f"SIGMA Engine complete: {len(final):,} retained points, "
        f"{len(centers):,} clusters, {len(partitions):,} partitions"
    )

    return EngineResult(final, centers, partitions, paths, metadata)
