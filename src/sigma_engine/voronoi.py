"""Network Voronoi ownership and its explicit 2-D SIGMA surface approximation.

The exact Voronoi object lives on the one-dimensional road graph.  Spatial overlap between
*types*, however, needs polygons.  SIGMA therefore renders the network ownership onto a
regular boundary grid: each cell representative point attaches to its nearest *reachable*
road location for that type and inherits the cluster owning that road location.

Road components containing no retained point of the current type are excluded from this
2-D attachment step.  Network distance to that type is undefined on those components, so
using them as anchors would create silent holes in the map.  The reachable-subnetwork rule
instead produces a complete boundary surface while keeping every ownership decision tied to
a road component on which network distance to a source is defined.

Algorithmic lineage: the exact network ownership concept follows network
Voronoi diagrams in Okabe et al. (2000), Hakimi, Labbe & Schmeichel (1992),
Erwig (2000), https://doi.org/10.1002/1097-0037(200010)36:3%3C156::AID-NET2%3E3.0.CO;2-L,
and Okabe et al. (2008).  Multi-source shortest paths use Dijkstra (1959) via
SciPy (Virtanen et al., 2020).  The 2-D boundary-grid surface is a SIGMA
rendering approximation of network ownership, not an exact planar Voronoi
diagram.  The ordinary Euclidean Voronoi path is a clearly labeled recovery
fallback after the network attempts fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse.csgraph import dijkstra

from .network import AugmentedNetwork


@dataclass(frozen=True)
class NetworkVoronoi:
    """Nearest-cluster distance and owner for every augmented road node."""

    node_distance: np.ndarray
    node_cluster: np.ndarray


def _tol(*values):
    """Scale-aware distance comparison tolerance used by optimized Voronoi ties."""
    scale = np.asarray(1.0)
    for value in values:
        a = np.abs(np.asarray(value, dtype=float))
        scale = np.maximum(scale, np.where(np.isfinite(a), a, 0.0))
    return np.maximum(1e-9, 1e-12 * scale)


def _tie_candidates(network: AugmentedNetwork, dist: np.ndarray, owner: np.ndarray) -> set[int]:
    """Safe superset of clusters that may win a shortest-path tie at a node."""
    finite = dist[np.isfinite(dist)]
    largest = float(finite.max()) if finite.size else 0.0
    candidate_t = float(_tol(largest))
    u = network.edges["u"].to_numpy(np.int64, copy=False)
    v = network.edges["v"].to_numpy(np.int64, copy=False)
    w = network.edges["length"].to_numpy(float, copy=False)
    reachable = np.isfinite(dist[u]) & np.isfinite(dist[v])
    u, v, w = u[reachable], v[reachable], w[reachable]
    du, dv = dist[u], dist[v]
    tight = (du + w - dv <= 8.0 * candidate_t) | (dv + w - du <= 8.0 * candidate_t)
    differ = tight & (owner[u] != owner[v])
    values = set(owner[u[differ]].tolist()) | set(owner[v[differ]].tolist())
    return {int(value) for value in values if int(value) >= 0}


def network_voronoi(
    network: AugmentedNetwork,
    source_nodes: np.ndarray,
    source_clusters: np.ndarray,
) -> NetworkVoronoi:
    """Nearest-cluster ownership using optimized SciPy multi-source Dijkstra."""
    source_nodes = np.asarray(source_nodes, dtype=np.int64)
    source_clusters = np.asarray(source_clusters, dtype=np.int64)
    if source_nodes.ndim != 1 or source_clusters.ndim != 1:
        raise ValueError("Voronoi source nodes/clusters must be one-dimensional")
    if len(source_nodes) != len(source_clusters):
        raise ValueError("Voronoi source nodes and cluster labels are misaligned")
    if len(source_nodes) == 0:
        raise ValueError("network Voronoi requires at least one source")
    if np.any(source_clusters < 0):
        raise ValueError("network Voronoi sources must not include noise cluster -1")
    n = network.sparse_adjacency.shape[0]
    if source_nodes.min() < 0 or source_nodes.max() >= n:
        raise ValueError("network Voronoi received an unknown source node")

    sentinel = np.iinfo(np.int64).max
    cluster_at_node = np.full(n, sentinel, dtype=np.int64)
    np.minimum.at(cluster_at_node, source_nodes, source_clusters)
    source = np.flatnonzero(cluster_at_node != sentinel)

    dist, _, nearest = dijkstra(
        network.sparse_adjacency,
        directed=True,
        indices=source,
        min_only=True,
        return_predecessors=True,
    )
    dist = np.asarray(dist, dtype=float)
    nearest = np.asarray(nearest, dtype=np.int64)
    reachable = np.isfinite(dist)
    owner = np.full(n, -1, dtype=np.int64)
    owner[reachable] = cluster_at_node[nearest[reachable]]

    finite = dist[reachable]
    largest = float(finite.max()) if finite.size else 0.0
    candidate_t = float(_tol(largest))
    for cluster in sorted(_tie_candidates(network, dist, owner)):
        cluster_sources = np.unique(source_nodes[source_clusters == cluster])
        dist_c = np.asarray(
            dijkstra(
                network.sparse_adjacency,
                directed=True,
                indices=cluster_sources,
                min_only=True,
                limit=largest + 2.0 * candidate_t,
            ),
            dtype=float,
        )
        near = np.flatnonzero(np.isfinite(dist_c) & (dist_c <= dist + 2.0 * candidate_t))
        if len(near) == 0:
            continue
        local_t = _tol(dist_c[near], dist[near])
        ties = near[dist_c[near] <= dist[near] + local_t]
        replace = ties[(owner[ties] < 0) | (cluster < owner[ties])]
        owner[replace] = int(cluster)

    return NetworkVoronoi(dist, owner)


def _boundary_geometry(boundary: gpd.GeoDataFrame, crs) -> object:
    """Return one valid polygonal boundary in the road CRS."""
    if boundary.crs is None:
        raise ValueError("boundary CRS is missing")
    if boundary.empty:
        raise ValueError("boundary is empty")

    frame = boundary.to_crs(crs) if boundary.crs != crs else boundary
    geometries = [
        geometry
        for geometry in frame.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        raise ValueError("boundary is empty")

    coordinates = shapely.get_coordinates(np.asarray(geometries, dtype=object))
    if coordinates.size and not np.isfinite(coordinates[:, :2]).all():
        raise ValueError("boundary contains non-finite coordinates")

    merged = shapely.union_all(np.asarray(geometries, dtype=object))
    if merged.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("boundary must be polygonal")
    if not shapely.is_valid(merged):
        raise ValueError(f"boundary geometry is invalid: {shapely.is_valid_reason(merged)}")
    return merged


def _grid(boundary, resolution: float, max_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized boundary-clipped grid using shared coordinate lines."""
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Voronoi surface resolution must be finite and positive")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1:
        raise ValueError("Voronoi max_cells must be a positive integer")
    x_min, y_min, x_max, y_max = boundary.bounds
    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    candidate_count = nx * ny
    if candidate_count > max_cells:
        raise ValueError(
            f"Voronoi surface requires {candidate_count:,} candidate cells, above "
            f"max_cells={max_cells:,}; use a coarser resolution or raise the guard explicitly"
        )
    xs = x_min + np.arange(nx + 1, dtype=float) * resolution
    ys = y_min + np.arange(ny + 1, dtype=float) * resolution
    col, row = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    col, row = col.ravel(), row.ravel()
    squares = shapely.box(xs[col], ys[row], xs[col + 1], ys[row + 1])
    intersects = shapely.intersects(squares, boundary)
    clipped = shapely.intersection(squares[intersects], boundary)
    positive = shapely.area(clipped) > 0
    clipped = clipped[positive]
    if len(clipped) == 0:
        raise ValueError("Voronoi surface grid produced no positive-area boundary cells")
    return clipped, shapely.point_on_surface(clipped)


def _owner_on_edge(
    result: NetworkVoronoi,
    u: int,
    v: int,
    length: float,
    offset: float,
) -> int | None:
    owner_u = int(result.node_cluster[u])
    owner_v = int(result.node_cluster[v])
    tolerance = max(1e-10, length * 1e-12)
    if owner_u < 0 and owner_v < 0:
        return None
    if offset <= tolerance:
        return None if owner_u < 0 else owner_u
    if length - offset <= tolerance:
        return None if owner_v < 0 else owner_v
    if owner_u == owner_v:
        return None if owner_u < 0 else owner_u
    distance_u = result.node_distance[u]
    distance_v = result.node_distance[v]
    if not (np.isfinite(distance_u) and np.isfinite(distance_v)):
        value = owner_u if np.isfinite(distance_u) else owner_v
        return None if value < 0 else value
    split = float(np.clip((distance_v + length - distance_u) / 2.0, 0.0, length))
    if abs(offset - split) <= tolerance:
        choices = [owner for owner in (owner_u, owner_v) if owner >= 0]
        return min(choices) if choices else None
    value = owner_u if offset < split else owner_v
    return None if value < 0 else value


def _owner_pairs_vectorized(
    network: AugmentedNetwork,
    result: NetworkVoronoi,
    edge_positions: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Owner for aligned anchor/edge pairs, vectorized over every pair."""
    edges = network.edges.iloc[edge_positions]
    u = edges["u"].to_numpy(np.int64)
    v = edges["v"].to_numpy(np.int64)
    length = edges["length"].to_numpy(float)
    lines = edges.geometry.to_numpy()
    geometric = shapely.length(lines)
    measure = shapely.line_locate_point(lines, anchors)
    offset = np.divide(
        length * measure,
        geometric,
        out=np.zeros_like(length, dtype=float),
        where=geometric != 0,
    )
    tol = np.maximum(1e-10, length * 1e-12)
    offset = np.where(offset <= tol, 0.0, np.where(length - offset <= tol, length, offset))

    owner_u = result.node_cluster[u].astype(np.int64, copy=False)
    owner_v = result.node_cluster[v].astype(np.int64, copy=False)
    dist_u = result.node_distance[u]
    dist_v = result.node_distance[v]
    out = np.full(len(edge_positions), -1, dtype=np.int64)

    at_u = offset == 0.0
    at_v = ~at_u & (offset == length)
    out[at_u] = owner_u[at_u]
    out[at_v] = owner_v[at_v]
    inside = ~at_u & ~at_v
    same = inside & (owner_u == owner_v)
    out[same] = owner_u[same]

    split = inside & ~same
    both_missing = split & (owner_u < 0) & (owner_v < 0)
    finite_u = np.isfinite(dist_u)
    finite_v = np.isfinite(dist_v)
    only_u = split & finite_u & ~finite_v
    only_v = split & finite_v & ~finite_u
    out[only_u] = owner_u[only_u]
    out[only_v] = owner_v[only_v]

    active = split & ~both_missing & finite_u & finite_v
    if active.any():
        boundary = np.clip((dist_v[active] + length[active] - dist_u[active]) / 2.0, 0.0, length[active])
        o = offset[active]
        t = tol[active]
        left = owner_u[active]
        right = owner_v[active]
        chosen = np.where(o < boundary - t, left, np.where(o > boundary + t, right, np.minimum(left, right)))
        out[active] = chosen
    return out


def _owners_for_anchors(
    anchors: np.ndarray,
    network: AugmentedNetwork,
    result: NetworkVoronoi,
    reachable_edge_positions: np.ndarray,
) -> np.ndarray:
    """Vectorized nearest-road attachment, preserving SIGMA's tied-edge owner rule."""
    lines = network.edges.geometry.iloc[reachable_edge_positions].to_numpy()
    (point_index, local_edge), _ = shapely.STRtree(lines).query_nearest(
        anchors, all_matches=True, return_distance=True
    )
    if len(point_index) == 0:
        raise RuntimeError("failed to attach Voronoi surface anchors to the road network")
    edge_position = reachable_edge_positions[local_edge]
    owners = _owner_pairs_vectorized(network, result, edge_position, anchors[point_index])
    sentinel = np.iinfo(np.int64).max
    numeric = np.where(owners < 0, sentinel, owners).astype(np.int64, copy=False)
    chosen = np.full(len(anchors), sentinel, dtype=np.int64)
    np.minimum.at(chosen, point_index, numeric)
    if np.any(chosen == sentinel):
        raise RuntimeError("failed to assign one or more Voronoi surface cells")
    return chosen


def _empty_partitions(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "type": [],
            "cluster": [],
            "surface_area": [],
            "surface_resolution": [],
            "surface_method": [],
            "surface_seed_method": [],
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def surface_partition_for_type(
    network: AugmentedNetwork,
    type_points: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    resolution: float,
    max_cells: int = 1_000_000,
    progress: Callable[[str], None] | None = None,
) -> gpd.GeoDataFrame:
    """Render one economic type's network Voronoi as boundary-clipped polygons."""
    required = {"type", "cluster", "network_node"}
    missing = required - set(type_points.columns)
    if missing:
        raise ValueError(f"Voronoi type points are missing columns: {sorted(missing)}")
    if type_points.empty:
        return _empty_partitions(network.crs)
    if type_points["type"].nunique(dropna=False) != 1:
        raise ValueError("surface_partition_for_type received more than one economic type")

    result = network_voronoi(
        network,
        type_points["network_node"].to_numpy(np.int64, copy=False),
        type_points["cluster"].to_numpy(np.int64, copy=False),
    )
    boundary_geometry = _boundary_geometry(boundary, network.crs)
    cells, anchors = _grid(boundary_geometry, resolution, max_cells)
    type_value = str(type_points["type"].iloc[0])
    if progress is not None:
        progress(
            f"voronoi: type {type_value!r}: {len(type_points):,} retained points, "
            f"{len(cells):,} positive-area grid cells"
        )

    # Only road components reachable from at least one source of this type have a defined
    # network-Voronoi owner.  Excluding source-less components here prevents them from
    # creating unassigned holes in the 2-D surface.
    reachable_edge_positions = np.asarray(
        [
            position
            for position, edge in network.edges.iterrows()
            if np.isfinite(result.node_distance[int(edge.u)])
            and np.isfinite(result.node_distance[int(edge.v)])
        ],
        dtype=np.int64,
    )
    if reachable_edge_positions.size == 0:
        raise RuntimeError("network Voronoi has no source-reachable road edges")

    if progress is not None:
        progress(f"voronoi: type {type_value!r}: vectorized assignment of {len(anchors):,} cells")
    labels = _owners_for_anchors(anchors, network, result, reachable_edge_positions)
    if progress is not None:
        progress(f"voronoi: type {type_value!r}: assigned {len(anchors):,}/{len(anchors):,} cells")

    frame = gpd.GeoDataFrame(
        {"cluster": labels.astype(np.int64)},
        geometry=gpd.GeoSeries(cells, crs=network.crs),
        crs=network.crs,
    )
    frame["type"] = type_value

    dissolved = frame.dissolve(by=["type", "cluster"], as_index=False)
    expected_clusters = set(type_points["cluster"].astype(int))
    actual_clusters = set(dissolved["cluster"].astype(int))
    missing_clusters = sorted(expected_clusters - actual_clusters)
    if missing_clusters:
        raise ValueError(
            f"Voronoi surface resolution {resolution:g} is too coarse for type {type_value!r}; "
            f"clusters {missing_clusters[:20]} received no grid cell. Use a finer resolution."
        )

    dissolved["surface_area"] = dissolved.geometry.area.astype(float)
    dissolved["surface_resolution"] = float(resolution)
    dissolved["surface_method"] = "network_voronoi"
    dissolved["surface_seed_method"] = "cluster_member_network_sources"
    return dissolved[
        [
            "type",
            "cluster",
            "surface_area",
            "surface_resolution",
            "surface_method",
            "surface_seed_method",
            "geometry",
        ]
    ]


def _raw_cluster_centroid(type_points: gpd.GeoDataFrame, cluster: int) -> object:
    """Projected arithmetic centroid of the retained observations in one cluster."""
    group = type_points.loc[type_points["cluster"].astype(int) == int(cluster)]
    if group.empty:
        raise ValueError(f"cluster {cluster} has no retained points for Euclidean Voronoi fallback")
    xy = shapely.get_coordinates(group.geometry.to_numpy())
    if len(xy) != len(group) or not np.isfinite(xy[:, :2]).all():
        raise ValueError(
            f"cluster {cluster} has invalid Point geometry for Euclidean Voronoi fallback"
        )
    mean = xy[:, :2].mean(axis=0)
    return shapely.Point(float(mean[0]), float(mean[1]))


def euclidean_surface_partition_for_type(
    type_points: gpd.GeoDataFrame,
    type_centers: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    crs,
    progress: Callable[[str], None] | None = None,
) -> gpd.GeoDataFrame:
    """Planar Voronoi fallback using one final center coordinate per cluster.

    Coincident center sites cannot define distinct planar Voronoi cells.  For only those
    clusters, the raw projected centroid of their retained observations replaces the center
    site.  If sites remain coincident, or clipping leaves a cluster without positive area,
    the fallback fails and the caller may skip the type.
    """
    required_points = {"type", "cluster"}
    missing = required_points - set(type_points.columns)
    if missing:
        raise ValueError(f"Euclidean Voronoi points are missing columns: {sorted(missing)}")
    required_centers = {"type", "cluster", "geometry"}
    missing = required_centers - set(type_centers.columns)
    if missing:
        raise ValueError(f"Euclidean Voronoi centers are missing columns: {sorted(missing)}")
    if type_points.empty:
        return _empty_partitions(crs)
    if type_points["type"].nunique(dropna=False) != 1:
        raise ValueError(
            "euclidean_surface_partition_for_type received more than one economic type"
        )

    type_value = str(type_points["type"].iloc[0])
    clusters = np.asarray(sorted(type_points["cluster"].astype(int).unique()), dtype=np.int64)
    centers = type_centers.copy()
    centers = centers.loc[centers["type"].astype(str) == type_value]
    if centers["cluster"].duplicated().any():
        raise ValueError(f"type {type_value!r} has duplicate cluster-center rows")
    centers = centers.set_index(centers["cluster"].astype(int), drop=False)
    missing_centers = [int(c) for c in clusters if int(c) not in centers.index]
    if missing_centers:
        raise ValueError(
            f"type {type_value!r} is missing centers for clusters {missing_centers[:20]}"
        )

    seed_geometry = [centers.loc[int(cluster), "geometry"] for cluster in clusters]
    seed_xy = shapely.get_coordinates(np.asarray(seed_geometry, dtype=object))
    if len(seed_xy) != len(clusters) or not np.isfinite(seed_xy[:, :2]).all():
        raise ValueError(f"type {type_value!r} has invalid center coordinates")
    seed_xy = seed_xy[:, :2].astype(float, copy=True)
    seed_method = np.full(len(clusters), "cluster_center", dtype=object)

    # Replace every member of an exactly coincident-center group with its own raw cluster
    # centroid.  This is deterministic and auditable; we never jitter sites silently.
    _, inverse, counts = np.unique(seed_xy, axis=0, return_inverse=True, return_counts=True)
    duplicate_rows = np.flatnonzero(counts[inverse] > 1)
    for row in duplicate_rows:
        centroid = _raw_cluster_centroid(type_points, int(clusters[row]))
        seed_xy[row] = (float(centroid.x), float(centroid.y))
        seed_method[row] = "raw_cluster_centroid_duplicate_center"

    unique_after = np.unique(seed_xy, axis=0)
    if len(unique_after) != len(seed_xy):
        duplicate_clusters = clusters[
            np.asarray(
                [sum(np.array_equal(xy, other) for other in seed_xy) > 1 for xy in seed_xy],
                dtype=bool,
            )
        ]
        raise ValueError(
            f"Euclidean Voronoi fallback for type {type_value!r} still has coincident sites "
            f"after centroid substitution; clusters {duplicate_clusters[:20].tolist()}"
        )

    boundary_geometry = _boundary_geometry(boundary, crs)
    if len(clusters) == 1:
        clipped = np.asarray([boundary_geometry], dtype=object)
    else:
        sites = shapely.points(seed_xy)
        multipoint = shapely.multipoints(seed_xy)
        collection = shapely.voronoi_polygons(
            multipoint,
            extend_to=shapely.envelope(boundary_geometry),
        )
        cells = np.asarray(shapely.get_parts(collection), dtype=object)
        if len(cells) != len(clusters):
            raise RuntimeError(
                f"Euclidean Voronoi fallback for type {type_value!r} produced "
                f"{len(cells)} cells for {len(clusters)} sites"
            )
        matches = shapely.STRtree(cells).query(sites, predicate="covered_by")
        assigned = np.full(len(clusters), -1, dtype=np.int64)
        if matches.size:
            # Exact boundary coincidences can yield multiple covering cells; choose the
            # smallest cell index deterministically.  Distinct sites still remain distinct.
            order = np.lexsort((matches[1], matches[0]))
            source = matches[0, order]
            cell = matches[1, order]
            first = np.ones(len(source), dtype=bool)
            first[1:] = source[1:] != source[:-1]
            assigned[source[first]] = cell[first]
        if (assigned < 0).any():
            missing_sites = clusters[assigned < 0].tolist()
            raise RuntimeError(
                f"Euclidean Voronoi fallback could not map sites to cells for "
                f"clusters {missing_sites[:20]}"
            )
        clipped = np.asarray(
            shapely.intersection(cells[assigned], boundary_geometry),
            dtype=object,
        )

    areas = np.asarray(shapely.area(clipped), dtype=float)
    boundary_area = float(shapely.area(boundary_geometry))
    area_tolerance = max(1e-8, boundary_area * 1e-9)
    missing_area = clusters[areas <= area_tolerance]
    if len(missing_area):
        raise ValueError(
            f"Euclidean Voronoi fallback for type {type_value!r} left clusters "
            f"{missing_area[:20].tolist()} without positive boundary area"
        )
    covered = shapely.union_all(clipped)
    uncovered_area = float(shapely.area(shapely.difference(boundary_geometry, covered)))
    overlap_error = abs(float(areas.sum()) - boundary_area)
    if uncovered_area > area_tolerance or overlap_error > max(area_tolerance, boundary_area * 1e-8):
        raise RuntimeError(
            f"Euclidean Voronoi fallback for type {type_value!r} did not form a complete "
            f"non-overlapping boundary partition (uncovered={uncovered_area:g}, "
            f"area_error={overlap_error:g})"
        )

    if progress is not None:
        progress(
            f"voronoi: type {type_value!r}: Euclidean fallback produced "
            f"{len(clusters):,} clipped partitions"
        )
    return gpd.GeoDataFrame(
        {
            "type": np.full(len(clusters), type_value, dtype=object),
            "cluster": clusters,
            "surface_area": areas,
            "surface_resolution": np.full(len(clusters), np.nan, dtype=float),
            "surface_method": np.full(len(clusters), "euclidean_voronoi_fallback", dtype=object),
            "surface_seed_method": seed_method,
        },
        geometry=gpd.GeoSeries(clipped, crs=crs),
        crs=crs,
    )


def all_surface_partitions(
    network: AugmentedNetwork,
    points: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    resolution: float,
    max_cells: int = 1_000_000,
    progress: Callable[[str], None] | None = None,
    *,
    centers: gpd.GeoDataFrame | None = None,
    auto_refine: bool = True,
    refine_factor: float = 0.5,
    max_refinements: int = 5,
    euclidean_fallback: bool = False,
    continue_on_error: bool = False,
    events: list[dict[str, object]] | None = None,
) -> gpd.GeoDataFrame:
    """Render partitions for every retained economic type with graceful fallback.

    Network Voronoi is attempted first.  A coarse grid is retried at progressively finer
    resolutions.  After those network attempts are exhausted, ordinary planar Voronoi is
    attempted using one final cluster center per site.  Only if that also fails is a type
    skipped in ``continue_on_error`` mode.
    """
    if not (0 < float(refine_factor) < 1):
        raise ValueError("Voronoi refine_factor must be between 0 and 1")
    if (
        isinstance(max_refinements, bool)
        or not isinstance(max_refinements, int)
        or max_refinements < 0
    ):
        raise ValueError("Voronoi max_refinements must be an integer >= 0")
    if euclidean_fallback and centers is None:
        raise ValueError("Euclidean Voronoi fallback requires cluster centers")

    parts: list[gpd.GeoDataFrame] = []
    groups = list(points.groupby("type", sort=True))
    total_types = len(groups)
    for position, (type_value, group) in enumerate(groups, start=1):
        type_text = str(type_value)
        if progress is not None:
            progress(f"voronoi: starting type {position:,}/{total_types:,} {type_text!r}")

        attempt_resolution = float(resolution)
        part = None
        last_error: Exception | None = None
        network_attempts = 0
        for attempt in range(max_refinements + 1):
            network_attempts = attempt + 1
            try:
                part = surface_partition_for_type(
                    network,
                    group,
                    boundary,
                    attempt_resolution,
                    max_cells,
                    progress=progress,
                )
                if attempt > 0:
                    if progress is not None:
                        progress(
                            f"voronoi: type {type_text!r}: recovered after {attempt} "
                            "refinement(s); "
                            f"using resolution={attempt_resolution:g}"
                        )
                    if events is not None:
                        events.append(
                            {
                                "stage": "voronoi",
                                "type": type_text,
                                "cluster": None,
                                "status": "recovered",
                                "action": "refined_resolution",
                                "attempt": attempt + 1,
                                "requested_resolution": float(resolution),
                                "used_resolution": float(attempt_resolution),
                                "error": str(last_error) if last_error is not None else None,
                            }
                        )
                break
            except (ValueError, RuntimeError, MemoryError) as exc:
                last_error = exc
                coarse = "too coarse" in str(exc).lower()
                can_refine = auto_refine and coarse and attempt < max_refinements
                if can_refine:
                    next_resolution = attempt_resolution * float(refine_factor)
                    if progress is not None:
                        progress(
                            f"voronoi: WARNING type {type_text!r}: {exc}; "
                            f"retrying at resolution={next_resolution:g}"
                        )
                    attempt_resolution = next_resolution
                    continue
                break

        if part is None and euclidean_fallback:
            if progress is not None:
                progress(
                    f"voronoi: WARNING type {type_text!r}: network Voronoi failed after "
                    f"{network_attempts} attempt(s); trying Euclidean Voronoi fallback"
                )
            try:
                assert centers is not None
                type_centers = centers.loc[centers["type"].astype(str) == type_text].copy()
                part = euclidean_surface_partition_for_type(
                    group,
                    type_centers,
                    boundary,
                    network.crs,
                    progress=progress,
                )
                substituted = sorted(
                    part.loc[
                        part["surface_seed_method"] == "raw_cluster_centroid_duplicate_center",
                        "cluster",
                    ].astype(int).tolist()
                )
                if substituted and events is not None:
                    events.append(
                        {
                            "stage": "voronoi",
                            "type": type_text,
                            "cluster": None,
                            "status": "recovered",
                            "action": "euclidean_voronoi_duplicate_centers_use_raw_centroids",
                            "attempt": network_attempts + 1,
                            "requested_resolution": float(resolution),
                            "used_resolution": None,
                            "error": (
                                "duplicate final center sites; substituted clusters "
                                f"{substituted[:20]}"
                            ),
                        }
                    )
                if events is not None:
                    events.append(
                        {
                            "stage": "voronoi",
                            "type": type_text,
                            "cluster": None,
                            "status": "recovered",
                            "action": "fallback_euclidean_voronoi",
                            "attempt": network_attempts + 1,
                            "requested_resolution": float(resolution),
                            "used_resolution": None,
                            "error": str(last_error) if last_error is not None else None,
                        }
                    )
                if progress is not None:
                    progress(
                        f"voronoi: type {type_text!r}: recovered with Euclidean Voronoi fallback"
                    )
            except (ValueError, RuntimeError, MemoryError) as fallback_exc:
                if not continue_on_error:
                    raise
                if progress is not None:
                    progress(
                        f"voronoi: WARNING type {type_text!r}: Euclidean Voronoi fallback "
                        f"also failed; skipping type: {fallback_exc}"
                    )
                if events is not None:
                    events.append(
                        {
                            "stage": "voronoi",
                            "type": type_text,
                            "cluster": None,
                            "status": "skipped",
                            "action": "skip_type_after_euclidean_voronoi_failure",
                            "attempt": network_attempts + 1,
                            "requested_resolution": float(resolution),
                            "used_resolution": None,
                            "error": f"network={last_error}; euclidean={fallback_exc}",
                        }
                    )
                part = None
        elif part is None:
            if not continue_on_error:
                assert last_error is not None
                raise last_error
            if progress is not None:
                progress(
                    f"voronoi: WARNING type {type_text!r} skipped after "
                    f"{network_attempts} network attempt(s): {last_error}"
                )
            if events is not None:
                events.append(
                    {
                        "stage": "voronoi",
                        "type": type_text,
                        "cluster": None,
                        "status": "skipped",
                        "action": "skip_type",
                        "attempt": network_attempts,
                        "requested_resolution": float(resolution),
                        "used_resolution": None,
                        "error": str(last_error),
                    }
                )

        if part is None:
            continue
        parts.append(part)
        if progress is not None:
            method = str(part["surface_method"].iloc[0]) if len(part) else "unknown"
            used = float(part["surface_resolution"].iloc[0]) if len(part) else np.nan
            detail = f"resolution={used:g}" if np.isfinite(used) else "planar fallback"
            progress(
                f"voronoi: type {type_text!r} complete ({len(part):,} partitions; "
                f"method={method}; {detail})"
            )

    if not parts:
        return _empty_partitions(network.crs)

    combined = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=network.crs)
