"""Exact network-node 1-median calculation using the shared SciPy road solver.

The primary objective is the classical network median/minisum problem of
Hakimi (1964), https://doi.org/10.1287/opre.12.3.450; see also Hakimi (1965),
Kariv & Hakimi (1979), https://doi.org/10.1137/0137041, and Handler &
Mirchandani (1979).  Shortest-path distances are Dijkstra (1959) via SciPy.
SIGMA's Euclidean-centroid -> road-snap -> node procedure is explicitly a
recovery fallback and is not presented as an exact network 1-median.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point

from ._sparse_solver import SparseRoadSolver
from .network import AugmentedNetwork


@dataclass(frozen=True)
class NetworkMedian:
    network_node: int
    objective: float
    geometry: Point


def network_1_median(
    network: AugmentedNetwork,
    demand_nodes: np.ndarray,
    solver: SparseRoadSolver | None = None,
) -> NetworkMedian:
    """Solve the exact unit-weight 1-median by a batched SciPy demand-by-node reduction."""
    solver = solver or SparseRoadSolver.from_augmented(network)
    best_node, objective = solver.one_median(demand_nodes)
    geometry = network.nodes.geometry.iloc[best_node]
    return NetworkMedian(best_node, objective, geometry)


def _euclidean_centroid_snap_to_network_node(
    group: gpd.GeoDataFrame,
    network: AugmentedNetwork,
    solver: SparseRoadSolver,
    edge_cache: dict[int, tuple[np.ndarray, object]],
) -> dict[str, object]:
    """Fallback center: Euclidean centroid -> nearest road -> nearest node on that road.

    The nearest-road search is restricted to the road component containing the cluster.
    SIGMA's downstream distance and Voronoi code require an actual network-node ID, so when
    the continuous road snap lies inside an edge, the closer endpoint of that already-split
    augmented edge is used (smallest node ID breaks exact ties).
    """
    demand_nodes = group["network_node"].to_numpy(np.int64, copy=False)
    components = np.unique(solver.component[demand_nodes])
    if len(components) != 1:
        raise ValueError("centroid fallback requires all cluster points on one road component")
    component = int(components[0])

    xy = shapely.get_coordinates(group.geometry.to_numpy())
    if len(xy) != len(group):
        raise ValueError("centroid fallback requires non-empty Point geometries")
    centroid_xy = xy.mean(axis=0)
    centroid = Point(float(centroid_xy[0]), float(centroid_xy[1]))

    if component not in edge_cache:
        edge_u = network.edges["u"].to_numpy(np.int64, copy=False)
        eligible = np.flatnonzero(solver.component[edge_u] == component).astype(np.int64)
        if len(eligible) == 0:
            raise RuntimeError("cluster road component contains no usable road edges")
        edge_geometries = network.edges.geometry.to_numpy()[eligible]
        edge_cache[component] = (eligible, shapely.STRtree(edge_geometries))
    eligible, tree = edge_cache[component]

    nearest_local = np.asarray(tree.query_nearest(centroid, all_matches=True), dtype=np.int64)
    if nearest_local.size == 0:
        raise RuntimeError("centroid fallback could not find a road edge")
    candidate_rows = eligible[nearest_local.reshape(-1)]
    edge_row = int(candidate_rows.min())
    edge = network.edges.iloc[edge_row]
    line = edge.geometry
    snapped = line.interpolate(line.project(centroid))

    u, v = int(edge["u"]), int(edge["v"])
    node_xy = shapely.get_coordinates(network.nodes.geometry.iloc[[u, v]].to_numpy())
    sx, sy = float(snapped.x), float(snapped.y)
    endpoint_distance = np.hypot(node_xy[:, 0] - sx, node_xy[:, 1] - sy)
    best = float(endpoint_distance.min())
    tolerance = max(1e-12, abs(best) * 1e-12)
    candidates = np.asarray([u, v], dtype=np.int64)[endpoint_distance <= best + tolerance]
    node = int(candidates.min())

    euclidean_objective = float(np.hypot(xy[:, 0] - centroid.x, xy[:, 1] - centroid.y).sum())
    return {
        "network_node": node,
        "geometry": network.nodes.geometry.iloc[node],
        "euclidean_objective": euclidean_objective,
        "centroid_x": float(centroid.x),
        "centroid_y": float(centroid.y),
        "road_snap_x": sx,
        "road_snap_y": sy,
        "centroid_to_road_distance": float(centroid.distance(snapped)),
        "road_snap_to_node_distance": float(best),
    }


def cluster_centers(
    points: gpd.GeoDataFrame,
    network: AugmentedNetwork,
    progress: Callable[[str], None] | None = None,
    *,
    continue_on_error: bool = False,
    events: list[dict[str, object]] | None = None,
) -> gpd.GeoDataFrame:
    required = {"type", "cluster", "network_node"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"cluster-center input is missing columns: {sorted(missing)}")
    if points.empty:
        raise ValueError("cluster-center calculation requires at least one retained point")
    if (points["cluster"] < 0).any():
        raise ValueError("cluster-center calculation received HDBSCAN noise rows")

    solver = SparseRoadSolver.from_augmented(network)
    edge_cache: dict[int, tuple[np.ndarray, object]] = {}
    rows: list[dict[str, object]] = []
    groups = list(points.groupby(["type", "cluster"], sort=True))
    total = len(groups)
    step = max(1, total // 20)
    for position, ((type_value, cluster), group) in enumerate(groups, start=1):
        if progress is not None and (position == 1 or position % step == 0 or position == total):
            progress(
                f"centers: solving {position:,}/{total:,} "
                f"[{str(type_value)}, {int(cluster)}] ({len(group):,} demand points)"
            )
        center_row: dict[str, object]
        try:
            solved = network_1_median(
                network,
                group["network_node"].to_numpy(np.int64, copy=False),
                solver,
            )
            center_row = {
                "type": str(type_value),
                "cluster": int(cluster),
                "center_network_node": solved.network_node,
                "median_objective": solved.objective,
                "center_method": "network_1_median",
                "fallback_euclidean_objective": np.nan,
                "fallback_centroid_x": np.nan,
                "fallback_centroid_y": np.nan,
                "fallback_road_snap_x": np.nan,
                "fallback_road_snap_y": np.nan,
                "fallback_centroid_to_road_distance": np.nan,
                "fallback_road_snap_to_node_distance": np.nan,
                "geometry": solved.geometry,
            }
        except (ValueError, RuntimeError, MemoryError) as exc:
            if not continue_on_error:
                raise
            if progress is not None:
                progress(
                    f"centers: WARNING network 1-median failed for "
                    f"[{str(type_value)}, {int(cluster)}]: {exc}; trying Euclidean centroid fallback"
                )
            try:
                fallback = _euclidean_centroid_snap_to_network_node(
                    group, network, solver, edge_cache
                )
            except (ValueError, RuntimeError, MemoryError) as fallback_exc:
                if progress is not None:
                    progress(
                        f"centers: WARNING centroid fallback also failed for "
                        f"[{str(type_value)}, {int(cluster)}]; skipping cluster: {fallback_exc}"
                    )
                if events is not None:
                    events.append(
                        {
                            "stage": "center",
                            "type": str(type_value),
                            "cluster": int(cluster),
                            "status": "skipped",
                            "action": "skip_cluster_after_centroid_fallback_failure",
                            "attempt": 2,
                            "requested_resolution": None,
                            "used_resolution": None,
                            "error": f"network={exc}; centroid={fallback_exc}",
                        }
                    )
                continue
            center_row = {
                "type": str(type_value),
                "cluster": int(cluster),
                "center_network_node": int(fallback["network_node"]),
                "median_objective": np.nan,
                "center_method": "euclidean_centroid_road_snap_fallback",
                "fallback_euclidean_objective": fallback["euclidean_objective"],
                "fallback_centroid_x": fallback["centroid_x"],
                "fallback_centroid_y": fallback["centroid_y"],
                "fallback_road_snap_x": fallback["road_snap_x"],
                "fallback_road_snap_y": fallback["road_snap_y"],
                "fallback_centroid_to_road_distance": fallback["centroid_to_road_distance"],
                "fallback_road_snap_to_node_distance": fallback["road_snap_to_node_distance"],
                "geometry": fallback["geometry"],
            }
            if events is not None:
                events.append(
                    {
                        "stage": "center",
                        "type": str(type_value),
                        "cluster": int(cluster),
                        "status": "recovered",
                        "action": "fallback_euclidean_centroid_road_snap",
                        "attempt": 2,
                        "requested_resolution": None,
                        "used_resolution": None,
                        "error": str(exc),
                    }
                )
            if progress is not None:
                progress(
                    f"centers: [{str(type_value)}, {int(cluster)}] recovered with "
                    "Euclidean centroid -> road snap fallback"
                )
        rows.append(center_row)
    if progress is not None:
        progress(
            f"centers: {len(rows):,}/{total:,} network 1-medians complete"
            + (f"; {total - len(rows):,} skipped" if len(rows) != total else "")
        )
    if not rows:
        return gpd.GeoDataFrame(
            {
                "type": [],
                "cluster": [],
                "center_network_node": [],
                "median_objective": [],
                "center_method": [],
                "fallback_euclidean_objective": [],
                "fallback_centroid_x": [],
                "fallback_centroid_y": [],
                "fallback_road_snap_x": [],
                "fallback_road_snap_y": [],
                "fallback_centroid_to_road_distance": [],
                "fallback_road_snap_to_node_distance": [],
            },
            geometry=gpd.GeoSeries([], crs=points.crs),
            crs=points.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=points.crs)
