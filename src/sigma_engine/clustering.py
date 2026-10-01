"""Sparse road-network HDBSCAN*, independently by economic type.

The clustering backend uses bounded shortest-path neighbour discovery on the road graph and
runs HDBSCAN* directly on the resulting sparse distance graph. It never materialises an
``n x n`` all-pairs distance matrix.

Algorithmic lineage: HDBSCAN* follows Campello, Moulavi & Sander (2013) and
Campello et al. (2015); clustering in network space follows Yiu & Mamoulis
(2004), https://doi.org/10.1145/1007568.1007619; nearest-segment snapping and
network-space density clustering are also discussed by Wang et al. (2019),
https://doi.org/10.3390/ijgi8050218.  Bounded neighborhood search has a
precedent in OPTICS (Ankerst et al., 1999), while road shortest paths use
Dijkstra (1959) through SciPy (Virtanen et al., 2020).  SIGMA's adaptive-radius
stopping diagnostic and Euclidean recovery path are engineering/fallback rules,
not parts of the published HDBSCAN* model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import geopandas as gpd
import numpy as np
import shapely
from sklearn.cluster import HDBSCAN as EuclideanHDBSCAN

from ._sparse_hdbscan import hdbscan
from ._sparse_network import (
    DEFAULT_MAX_NEIGHBOR_PAIRS,
    NeighborPairLimitError,
    NetworkGraph,
    Snaps,
    build_network_graph,
    distinct_positions,
    neighbor_graph,
    snap_points,
)
from .network import AugmentedNetwork

_MAX_START_REDUCTIONS = 12


@dataclass(frozen=True)
class ClusteringConfig:
    """Parameters for sparse shortest-path HDBSCAN*."""

    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    max_distance: float = 5_000.0
    distance_mode: str = "adaptive"
    min_distance: float | None = None
    distance_growth: float = 1.5
    distance_steps: int = 4
    core_truncation_tolerance: float = 0.01
    stability_tolerance: float = 1e-12
    max_neighbor_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS

    def validate(self) -> None:
        if (
            isinstance(self.min_cluster_size, bool)
            or not isinstance(self.min_cluster_size, (int, np.integer))
            or self.min_cluster_size < 2
        ):
            raise ValueError("min_cluster_size must be an integer >= 2")
        if self.min_samples is not None and (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, (int, np.integer))
            or self.min_samples < 1
        ):
            raise ValueError("min_samples must be None or an integer >= 1")
        if self.cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
        if self.distance_mode not in {"fixed", "adaptive"}:
            raise ValueError("distance_mode must be 'fixed' or 'adaptive'")
        if not np.isfinite(self.max_distance) or self.max_distance <= 0:
            raise ValueError("max_distance must be finite and positive")
        if self.min_distance is not None and (
            not np.isfinite(self.min_distance)
            or self.min_distance <= 0
            or self.min_distance > self.max_distance
        ):
            raise ValueError("min_distance must be None or in (0, max_distance]")
        if not np.isfinite(self.distance_growth) or self.distance_growth <= 1:
            raise ValueError("distance_growth must be finite and > 1")
        if (
            isinstance(self.distance_steps, bool)
            or not isinstance(self.distance_steps, (int, np.integer))
            or self.distance_steps < 2
        ):
            raise ValueError("distance_steps must be an integer >= 2")
        if not np.isfinite(self.core_truncation_tolerance) or not (
            0 <= self.core_truncation_tolerance <= 1
        ):
            raise ValueError("core_truncation_tolerance must be between 0 and 1")
        if not np.isfinite(self.stability_tolerance) or self.stability_tolerance < 0:
            raise ValueError("stability_tolerance must be finite and >= 0")
        if (
            isinstance(self.max_neighbor_pairs, bool)
            or not isinstance(self.max_neighbor_pairs, (int, np.integer))
            or self.max_neighbor_pairs < 1
        ):
            raise ValueError("max_neighbor_pairs must be a positive integer")


@dataclass(frozen=True)
class SparseClusteringContext:
    """Road graph and point snaps reused across every economic type."""

    graph: NetworkGraph
    snaps: Snaps
    edge_id: np.ndarray


@dataclass(frozen=True)
class _SparseRun:
    labels: np.ndarray
    membership: np.ndarray
    summary: dict[str, object]
    trace: list[dict[str, object]]


def _run_euclidean_hdbscan_by_component(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph,
    config: ClusteringConfig,
    *,
    label: str,
    progress: Callable[[str], None] | None,
) -> _SparseRun:
    """Fallback HDBSCAN on projected XY coordinates, separately by road component.

    The fallback deliberately preserves road-component separation.  A plain Euclidean run
    over the whole type could join observations that are close in the plane but unreachable
    on the road graph, which would create downstream clusters without a valid network center.
    """
    xy = shapely.get_coordinates(points.geometry.to_numpy())
    if len(xy) != len(points):
        raise ValueError("Euclidean HDBSCAN fallback requires non-empty Point geometries")

    component = graph.component[snaps.u].astype(np.int64)
    labels = np.full(len(points), -1, dtype=np.int64)
    membership = np.zeros(len(points), dtype=float)
    next_label = 0

    for comp in np.unique(component):
        local = np.flatnonzero(component == comp)
        if len(local) < config.min_cluster_size:
            continue
        model = EuclideanHDBSCAN(
            min_cluster_size=int(config.min_cluster_size),
            min_samples=None if config.min_samples is None else int(config.min_samples),
            metric="euclidean",
            cluster_selection_method=config.cluster_selection_method,
            allow_single_cluster=bool(config.allow_single_cluster),
            copy=True,
        ).fit(xy[local])
        local_labels = np.asarray(model.labels_, dtype=np.int64)
        membership[local] = np.asarray(model.probabilities_, dtype=float)
        for raw in sorted(set(local_labels.tolist()) - {-1, -2, -3}):
            members = local[local_labels == raw]
            labels[members] = next_label
            next_label += 1

    noise_share = float((labels < 0).mean()) if len(labels) else 0.0
    if progress is not None:
        progress(
            f"clustering: {label}: Euclidean HDBSCAN fallback complete: "
            f"{next_label:,} clusters, {(labels >= 0).sum():,}/{len(labels):,} retained"
        )
    return _SparseRun(
        labels=labels,
        membership=membership,
        summary={
            "clustering_method": "euclidean_hdbscan_fallback",
            "n_points": int(len(points)),
            "n_positions": int(len(points)),
            "n_pairs": 0,
            "n_clusters": int(next_label),
            "noise_share": noise_share,
            "share_core_structurally_unreachable": None,
            "share_core_truncated": None,
            "max_distance": 0.0,
            "requested_max_distance": float(config.max_distance),
            "distance_mode": "euclidean_fallback",
            "distance_status": "fallback_euclidean_hdbscan",
            "distance_steps_run": 0,
            "distance_stable": None,
            "pair_limit_encountered": False,
            "pair_limit_distance": None,
            "core_truncation_tolerance": float(config.core_truncation_tolerance),
        },
        trace=[],
    )


def prepare_sparse_context(
    roads: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    *,
    vertex_digits: int | None = 11,
    progress: Callable[[str], None] | None = None,
) -> SparseClusteringContext:
    """Build the optimized road graph once and snap all classified points vectorially."""
    if roads.crs is None:
        raise ValueError("road network CRS is missing")
    if getattr(roads.crs, "is_geographic", False):
        raise ValueError("road network must use a projected CRS with linear units")
    if points.crs is None:
        raise ValueError("point CRS is missing")
    projected = points.to_crs(roads.crs) if points.crs != roads.crs else points
    if progress is not None:
        progress(f"roads: building sparse SciPy/Shapely graph from {len(roads):,} line features")
    graph = build_network_graph(roads.geometry.to_numpy(), vertex_digits=vertex_digits)
    if progress is not None:
        progress(
            f"roads: sparse graph ready: {graph.n_vertices:,} vertices, "
            f"{graph.n_arcs:,} arcs, {graph.n_components:,} components"
        )
        progress(f"road snap: vectorized snap of {len(projected):,} classified points")
    xy = shapely.get_coordinates(projected.geometry.to_numpy())
    if len(xy) != len(projected):
        raise ValueError("all clustering geometries must be non-empty Points")
    snaps = snap_points(graph, xy)

    # Arc arrays are lexicographically sorted by construction. Map each snapped (u,v)
    # pair back to that stable arc row without a Python dictionary over points.
    keys = graph.arc_u.astype(np.int64) * np.int64(graph.n_vertices) + graph.arc_v.astype(
        np.int64
    )
    point_keys = snaps.u.astype(np.int64) * np.int64(graph.n_vertices) + snaps.v.astype(
        np.int64
    )
    edge_id = np.searchsorted(keys, point_keys).astype(np.int64)
    if np.any(edge_id >= len(keys)) or np.any(keys[edge_id] != point_keys):
        raise RuntimeError("failed to map one or more snapped points to a road arc")
    if progress is not None:
        progress(
            f"road snap: complete; max snap distance="
            f"{float(snaps.snap_distance.max(initial=0.0)):g}"
        )
    return SparseClusteringContext(graph=graph, snaps=snaps, edge_id=edge_id)


def _start_radius(config: ClusteringConfig) -> float:
    if config.min_distance is not None:
        return float(config.min_distance)
    return float(config.max_distance) / (
        float(config.distance_growth) ** (int(config.distance_steps) - 1)
    )


def _trace_row(
    trial: int,
    radius: float,
    outcome: str,
    *,
    pair_count: int | None = None,
    n_clusters: int | None = None,
    noise_share: float | None = None,
    truncated_share: float | None = None,
    stable: bool | None = None,
    seconds: float,
) -> dict[str, object]:
    return {
        "trial": int(trial),
        "radius": float(radius),
        "outcome": outcome,
        "n_pairs": pair_count,
        "n_clusters": n_clusters,
        "noise_share": noise_share,
        "share_core_truncated": truncated_share,
        "same_as_previous": stable,
        "seconds": round(float(seconds), 3),
    }


def _run_sparse_hdbscan(
    graph: NetworkGraph,
    snaps: Snaps,
    config: ClusteringConfig,
    *,
    label: str,
    progress: Callable[[str], None] | None,
) -> _SparseRun:
    """Cluster one type and retain the adaptive-distance audit trail."""
    position, representative = distinct_positions(snaps)
    n_positions = len(representative)
    weights = np.bincount(position, minlength=n_positions).astype(np.int64)
    position_snaps = snaps.subset(representative)
    effective_min_samples = int(config.min_samples or config.min_cluster_size)
    total_weight = int(weights.sum())

    if len(snaps) < config.min_cluster_size or n_positions == 0:
        labels = np.full(len(snaps), -1, dtype=np.int64)
        return _SparseRun(
            labels=labels,
            membership=np.zeros(len(snaps), dtype=float),
            summary={
                "n_points": int(len(snaps)),
                "n_positions": int(n_positions),
                "n_pairs": 0,
                "n_clusters": 0,
                "noise_share": 1.0 if len(snaps) else 0.0,
                "share_core_structurally_unreachable": 1.0 if len(snaps) else 0.0,
                "share_core_truncated": 0.0,
                "max_distance": 0.0,
                "requested_max_distance": float(config.max_distance),
                "distance_mode": config.distance_mode,
                "distance_status": "insufficient_points",
                "distance_steps_run": 0,
                "distance_stable": None,
                "pair_limit_encountered": False,
                "pair_limit_distance": None,
                "core_truncation_tolerance": float(config.core_truncation_tolerance),
            },
            trace=[],
        )

    component = graph.component[position_snaps.u]
    component_weight = np.bincount(
        component, weights=weights, minlength=graph.n_components
    )
    structurally_unreachable = component_weight[component] < effective_min_samples
    structural_share = (
        float(weights[structurally_unreachable].sum()) / float(total_weight)
        if total_weight
        else 0.0
    )

    trace: list[dict[str, object]] = []
    ceiling = float(config.max_distance)
    radius = ceiling if config.distance_mode == "fixed" else _start_radius(config)
    previous_labels: np.ndarray | None = None
    previous_membership: np.ndarray | None = None
    last: tuple[np.ndarray, np.ndarray, float, int, float, int, float] | None = None
    last_stable = False
    successful_trials = 0
    trial = 0
    reductions = 0
    pair_limit_encountered = False
    pair_limit_distance: float | None = None
    smallest_failed = np.inf
    status = "fixed" if config.distance_mode == "fixed" else "ceiling_reached"

    while True:
        radius = min(ceiling, float(radius))
        if np.isclose(radius, ceiling, rtol=1e-12, atol=0.0):
            radius = ceiling
        if last is not None and radius >= smallest_failed * (1.0 - 1e-9):
            status = "pair_cap_limited"
            break

        trial += 1
        if progress is not None:
            progress(
                f"clustering: {label}: trial {trial}, radius={radius:g}; "
                f"{len(snaps):,} observations -> {n_positions:,} distinct network positions"
            )
        began = time.perf_counter()
        try:
            distances = neighbor_graph(
                graph,
                position_snaps,
                max_distance=radius,
                max_pairs=config.max_neighbor_pairs,
            )
        except NeighborPairLimitError as exc:
            pair_limit_encountered = True
            pair_limit_distance = float(exc.max_distance)
            smallest_failed = min(smallest_failed, radius)
            trace.append(
                _trace_row(
                    trial,
                    radius,
                    "pair_limit",
                    pair_count=int(exc.count),
                    seconds=time.perf_counter() - began,
                )
            )
            if last is not None:
                status = "pair_cap_limited"
                if progress is not None:
                    progress(
                        f"clustering: {label}: pair cap reached at radius={radius:g}; "
                        "using last successful adaptive trial"
                    )
                break
            if config.distance_mode != "adaptive" or config.min_distance is not None:
                raise
            smaller = radius / float(config.distance_growth)
            if smaller <= 0 or reductions >= _MAX_START_REDUCTIONS:
                raise
            reductions += 1
            radius = smaller
            continue

        result = hdbscan(
            distances,
            weights,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_method=config.cluster_selection_method,
            allow_single_cluster=config.allow_single_cluster,
        )
        obs_labels = result.labels[position]
        obs_membership = result.probabilities[position]
        pair_count = int(distances.nnz // 2)
        infinite_positions = np.isinf(result.core_distances)
        truncated_positions = infinite_positions & ~structurally_unreachable
        truncated_share = (
            float(weights[truncated_positions].sum()) / float(total_weight)
            if total_weight
            else 0.0
        )
        stable = (
            previous_labels is not None
            and np.array_equal(previous_labels, obs_labels)
            and np.allclose(
                previous_membership,
                obs_membership,
                atol=float(config.stability_tolerance),
                rtol=0.0,
                equal_nan=True,
            )
        )
        noise_share = float((obs_labels < 0).mean()) if len(obs_labels) else 0.0
        if progress is not None:
            progress(
                f"clustering: {label}: radius={radius:g} found {pair_count:,} neighbour pairs; "
                f"{result.n_clusters:,} clusters; core-truncated share={truncated_share:.3%}; "
                f"stable={'yes' if stable else 'no'}"
            )
        trace.append(
            _trace_row(
                trial,
                radius,
                "ok",
                pair_count=pair_count,
                n_clusters=int(result.n_clusters),
                noise_share=noise_share,
                truncated_share=truncated_share,
                stable=bool(stable) if previous_labels is not None else None,
                seconds=time.perf_counter() - began,
            )
        )
        successful_trials += 1
        last = (
            obs_labels.copy(),
            obs_membership.copy(),
            radius,
            pair_count,
            truncated_share,
            int(result.n_clusters),
            noise_share,
        )
        last_stable = bool(stable)

        if config.distance_mode == "fixed":
            status = "fixed"
            break
        if structural_share >= 1.0 - 1e-15 and truncated_share <= 1e-15:
            status = "structurally_unreachable"
            last_stable = True
            break
        if stable and truncated_share <= config.core_truncation_tolerance:
            status = "converged"
            break
        if radius >= ceiling:
            status = "ceiling_reached"
            break

        previous_labels = obs_labels.copy()
        previous_membership = obs_membership.copy()
        radius = min(ceiling, radius * float(config.distance_growth))

    if last is None:
        raise RuntimeError("sparse HDBSCAN produced no successful distance trial")
    labels, membership, final_radius, pair_count, truncated_share, n_clusters, noise_share = last
    summary = {
        "n_points": int(len(snaps)),
        "n_positions": int(n_positions),
        "n_pairs": int(pair_count),
        "n_clusters": int(n_clusters),
        "noise_share": float(noise_share),
        "share_core_structurally_unreachable": float(structural_share),
        "share_core_truncated": float(truncated_share),
        "max_distance": float(final_radius),
        "requested_max_distance": float(config.max_distance),
        "distance_mode": config.distance_mode,
        "distance_status": status,
        "distance_steps_run": int(successful_trials),
        "distance_stable": (
            None if config.distance_mode == "fixed" else bool(last_stable)
        ),
        "pair_limit_encountered": bool(pair_limit_encountered),
        "pair_limit_distance": pair_limit_distance,
        "core_truncation_tolerance": float(config.core_truncation_tolerance),
    }
    return _SparseRun(labels=labels, membership=membership, summary=summary, trace=trace)


def cluster_by_type(
    points: gpd.GeoDataFrame,
    context: SparseClusteringContext,
    config: ClusteringConfig | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    continue_on_error: bool = False,
    events: list[dict[str, object]] | None = None,
) -> gpd.GeoDataFrame:
    """Cluster every economic type using sparse bounded road-network distance.

    The returned GeoDataFrame carries two audit lists in ``.attrs``:
    ``clustering_summary`` (one final row per type) and ``clustering_trace``
    (one row per adaptive/fixed distance trial).
    """
    config = config or ClusteringConfig()
    config.validate()
    required = {"type", "canonical_id"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points are missing clustering columns: {sorted(missing)}")
    if points["type"].isna().any() or points["canonical_id"].isna().any():
        raise ValueError("type and canonical_id must be complete before clustering")
    if points["canonical_id"].astype(str).duplicated().any():
        raise ValueError("canonical_id must be unique before clustering")
    if isinstance(context, AugmentedNetwork):
        # Backward-compatible Python API. Production runs prepare the sparse context once.
        road_lines = gpd.GeoDataFrame(geometry=context.edges.geometry.copy(), crs=context.crs)
        context = prepare_sparse_context(road_lines, points, vertex_digits=None)
    if len(points) != len(context.snaps):
        raise ValueError("point rows and sparse snap rows are misaligned")

    result = points.copy()
    result["cluster"] = -1
    result["membership"] = 0.0
    result["network_component"] = context.graph.component[context.snaps.u].astype(np.int64)
    result["snap_distance_to_network"] = context.snaps.snap_distance.astype(float)
    result["snapped_edge_id"] = context.edge_id.astype(np.int64)

    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    groups = list(result.groupby("type", sort=True))
    for number, (type_value, rows) in enumerate(groups, start=1):
        ordered = rows.assign(_key=rows["canonical_id"].astype(str)).sort_values(
            "_key", kind="stable"
        )
        index = ordered.index.to_numpy(np.int64)
        subset = context.snaps.subset(index)
        type_text = str(type_value)
        if progress is not None:
            progress(
                f"clustering: type {number:,}/{len(groups):,} {type_text!r}, "
                f"{len(index):,} observations"
            )
        try:
            run = _run_sparse_hdbscan(
                context.graph,
                subset,
                config,
                label=f"type {type_text!r}",
                progress=progress,
            )
        except (ValueError, RuntimeError, MemoryError) as exc:
            if not continue_on_error:
                raise
            if progress is not None:
                progress(
                    f"clustering: WARNING network HDBSCAN failed for type {type_text!r}: "
                    f"{exc}; trying Euclidean HDBSCAN fallback"
                )
            try:
                run = _run_euclidean_hdbscan_by_component(
                    ordered,
                    subset,
                    context.graph,
                    config,
                    label=f"type {type_text!r}",
                    progress=progress,
                )
            except (ValueError, RuntimeError, MemoryError) as fallback_exc:
                if progress is not None:
                    progress(
                        f"clustering: WARNING Euclidean HDBSCAN fallback also failed for "
                        f"type {type_text!r}; skipping type: {fallback_exc}"
                    )
                if events is not None:
                    events.append(
                        {
                            "stage": "clustering",
                            "type": type_text,
                            "cluster": None,
                            "status": "skipped",
                            "action": "skip_type_after_euclidean_hdbscan_failure",
                            "attempt": 2,
                            "requested_resolution": None,
                            "used_resolution": None,
                            "error": f"network={exc}; euclidean={fallback_exc}",
                        }
                    )
                summary_rows.append(
                    {
                        "type": type_text,
                        "clustering_method": "failed",
                        "n_points": int(len(index)),
                        "n_positions": None,
                        "n_pairs": None,
                        "n_clusters": 0,
                        "noise_share": 1.0,
                        "share_core_structurally_unreachable": None,
                        "share_core_truncated": None,
                        "max_distance": 0.0,
                        "requested_max_distance": float(config.max_distance),
                        "distance_mode": config.distance_mode,
                        "distance_status": "failed_skipped",
                        "distance_steps_run": 0,
                        "distance_stable": None,
                        "pair_limit_encountered": False,
                        "pair_limit_distance": None,
                        "core_truncation_tolerance": float(config.core_truncation_tolerance),
                    }
                )
                continue
            if events is not None:
                events.append(
                    {
                        "stage": "clustering",
                        "type": type_text,
                        "cluster": None,
                        "status": "recovered",
                        "action": "fallback_euclidean_hdbscan",
                        "attempt": 2,
                        "requested_resolution": None,
                        "used_resolution": None,
                        "error": str(exc),
                    }
                )
        labels, membership = run.labels, run.membership

        # Normalize selected labels by smallest member canonical_id so [type, cluster]
        # keys stay stable even if hierarchy-internal numeric labels change.
        clusters: list[tuple[str, np.ndarray]] = []
        for raw in sorted(set(labels.tolist()) - {-1}):
            members_local = np.flatnonzero(labels == raw)
            members = index[members_local]
            first_id = min(str(v) for v in result.loc[members, "canonical_id"])
            clusters.append((first_id, members))
        clusters.sort(key=lambda item: item[0])
        for normalized, (_, members) in enumerate(clusters):
            result.loc[members, "cluster"] = normalized
        result.loc[index, "membership"] = membership

        retained = int(sum(len(m) for _, m in clusters))
        summary = {
            "type": type_text,
            "clustering_method": run.summary.get("clustering_method", "network_hdbscan"),
            **run.summary,
            "n_clusters": int(len(clusters)),
        }
        summary_rows.append(summary)
        trace_rows.extend({"type": type_text, **row} for row in run.trace)
        if progress is not None:
            if summary["clustering_method"] == "network_hdbscan":
                progress(
                    f"clustering: type {type_text!r} complete: {len(clusters):,} clusters, "
                    f"{retained:,}/{len(index):,} retained; radius={float(run.summary['max_distance']):g}, "
                    f"pairs={int(run.summary['n_pairs']):,}, "
                    f"status={run.summary['distance_status']}, "
                    f"truncated={float(run.summary['share_core_truncated']):.3%}"
                )
            else:
                progress(
                    f"clustering: type {type_text!r} complete via Euclidean fallback: "
                    f"{len(clusters):,} clusters, {retained:,}/{len(index):,} retained"
                )

    result["cluster"] = result["cluster"].astype(np.int64)
    result.attrs["clustering_summary"] = summary_rows
    result.attrs["clustering_trace"] = trace_rows
    return result


def retained_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "cluster" not in points:
        raise ValueError("cluster column is missing")
    retained = points.loc[points["cluster"] >= 0].copy()
    retained["_sparse_source_pos"] = retained.index.to_numpy(np.int64)
    return retained.reset_index(drop=True)
