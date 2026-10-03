"""Road-network construction, continuous point snapping, and shortest-path distance.

SIGMA uses one road graph throughout the workflow.  Points are first snapped to continuous
positions on the input linework; those positions are then inserted as graph vertices.  This
is important: all later clustering, medians, Voronoi ownership, and point-to-center distances
refer to the *same augmented graph* rather than re-snapping independently in each stage.

Topology rule
-------------
Consecutive coordinates of each input line become undirected graph arcs.  Two arcs connect
only when their canonicalized endpoint coordinates are identical.  A purely geometric
crossing therefore does not create a turn unless the source linework already contains a
shared vertex there.  This avoids inventing connections at bridges/flyovers.

Algorithmic lineage: locating observations on a spatial network and measuring
shortest-path distance follows Yiu & Mamoulis (2004),
https://doi.org/10.1145/1007568.1007619; nearest-segment snapping/splitting is
consistent with Wang et al. (2019), https://doi.org/10.3390/ijgi8050218.
Shortest paths are Dijkstra (1959), evaluated with SciPy (Virtanen et al.,
2020).  The shared-source-vertex topology rule, coordinate canonicalization,
and deterministic snapping ties are SIGMA engineering/modeling conventions.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from pathlib import Path
from typing import Callable

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.geometry import LineString, Point
from shapely.ops import substring


def _validate_vertex_digits(vertex_digits: int) -> int:
    """Validate coordinate-canonicalization precision.

    Nine decimal digits is intentionally conservative for projected coordinates.  The upper
    bound mainly catches accidental values (for example, passing a distance in metres into
    this argument) rather than imposing a mathematical requirement.
    """
    if isinstance(vertex_digits, bool) or not isinstance(vertex_digits, int):
        raise ValueError("vertex_digits must be an integer")
    if not 0 <= vertex_digits <= 15:
        raise ValueError("vertex_digits must be between 0 and 15")
    return vertex_digits


def _coord_key(x: float, y: float, digits: int) -> tuple[float, float]:
    """Canonical 2-D coordinate used as a stable graph-vertex key."""
    return (round(float(x), digits), round(float(y), digits))


def _require_projected_crs(gdf: gpd.GeoDataFrame) -> None:
    """Require a CRS in which line lengths are meaningful linear distances."""
    if gdf.crs is None:
        raise ValueError("road network CRS is missing")
    if getattr(gdf.crs, "is_geographic", False):
        raise ValueError("road network must use a projected CRS with linear units")


def _iter_lines(geometry) -> Iterable[LineString]:
    """Yield LineStrings from one lineal geometry, ignoring empty rows."""
    if geometry is None or geometry.is_empty:
        return
    kind = geometry.geom_type
    if kind == "LineString":
        yield geometry
    elif kind == "MultiLineString":
        yield from geometry.geoms
    else:
        raise ValueError(f"road geometry must be lineal, got {kind}")


@dataclass(frozen=True)
class RoadNetwork:
    """Immutable base road network before point positions are inserted."""

    crs: object
    graph: nx.Graph
    nodes: gpd.GeoDataFrame
    segments: gpd.GeoDataFrame
    vertex_digits: int

    @classmethod
    def from_geodataframe(cls, roads: gpd.GeoDataFrame, vertex_digits: int = 9) -> "RoadNetwork":
        """Build a deterministic undirected graph from line geometry.

        Each consecutive source-geometry coordinate pair becomes one straight segment.
        Duplicate/reversed copies of exactly the same canonical segment are collapsed; this
        cannot change shortest-path distance because their endpoints and geometry coincide.
        """
        vertex_digits = _validate_vertex_digits(vertex_digits)
        _require_projected_crs(roads)
        if roads.empty:
            raise ValueError("road network is empty")

        raw: list[tuple[tuple[float, float], tuple[float, float], LineString]] = []
        for row_pos, geometry in enumerate(roads.geometry):
            for line in _iter_lines(geometry):
                coordinates = np.asarray(line.coords, dtype=float)
                if coordinates.ndim != 2 or coordinates.shape[0] < 2:
                    continue
                if not np.isfinite(coordinates[:, :2]).all():
                    raise ValueError(f"road row {row_pos} contains non-finite coordinates")

                # SIGMA is a 2-D network model.  Any Z/M coordinate is intentionally ignored.
                for a, b in zip(coordinates[:-1], coordinates[1:], strict=True):
                    start = _coord_key(a[0], a[1], vertex_digits)
                    end = _coord_key(b[0], b[1], vertex_digits)
                    if start == end:
                        continue
                    # Canonical orientation makes every base segment independent of source
                    # digitization direction.  Without this, reversing one LineString could
                    # reverse snap offsets, change inserted-node numbering and alter a
                    # deterministic tie-break such as ``min(network_node)`` for a 1-median.
                    if end < start:
                        start, end = end, start
                    segment = LineString([start, end])
                    length = float(segment.length)
                    if not isfinite(length) or length <= 0:
                        continue
                    raw.append((start, end, segment))

        if not raw:
            raise ValueError("road network contains no positive-length segments")

        coordinate_keys = sorted({point for a, b, _ in raw for point in (a, b)})
        node_id = {xy: i for i, xy in enumerate(coordinate_keys)}

        graph = nx.Graph()
        for xy, network_node in node_id.items():
            graph.add_node(network_node, x=xy[0], y=xy[1])

        # The graph is simple rather than a MultiGraph.  At this stage every segment is a
        # straight line between its canonical endpoints, so reversed/duplicate endpoint
        # pairs describe the same geometric arc and can be safely collapsed.
        by_endpoints: dict[
            tuple[tuple[float, float], tuple[float, float]],
            tuple[tuple[float, float], tuple[float, float], LineString],
        ] = {}
        for start, end, geometry in raw:
            key = (min(start, end), max(start, end))
            incumbent = by_endpoints.get(key)
            candidate = (start, end, geometry)
            if incumbent is None or (float(geometry.length), geometry.wkb_hex) < (
                float(incumbent[2].length),
                incumbent[2].wkb_hex,
            ):
                by_endpoints[key] = candidate

        cooked = sorted(
            by_endpoints.values(),
            key=lambda item: (min(item[0], item[1]), max(item[0], item[1]), item[2].wkb_hex),
        )
        segment_rows: list[dict[str, object]] = []
        for edge_id, (start, end, geometry) in enumerate(cooked):
            u, v = node_id[start], node_id[end]
            length = float(geometry.length)
            segment_rows.append(
                {"edge_id": edge_id, "u": u, "v": v, "length": length, "geometry": geometry}
            )
            graph.add_edge(u, v, length=length, edge_id=edge_id)

        nodes = gpd.GeoDataFrame(
            {"network_node": np.arange(len(coordinate_keys), dtype=np.int64)},
            geometry=[Point(xy) for xy in coordinate_keys],
            crs=roads.crs,
        )
        segments = gpd.GeoDataFrame(segment_rows, geometry="geometry", crs=roads.crs)
        return cls(roads.crs, graph, nodes, segments, vertex_digits)


    @classmethod
    def from_sparse_graph(cls, sparse_graph, crs, vertex_digits: int = 9) -> "RoadNetwork":
        """Adapt the optimized SciPy/Shapely road graph without rebuilding line topology."""
        vertex_digits = _validate_vertex_digits(vertex_digits)
        xy = np.asarray(sparse_graph.vertex_xy, dtype=float)
        u = np.asarray(sparse_graph.arc_u, dtype=np.int64)
        v = np.asarray(sparse_graph.arc_v, dtype=np.int64)
        length = np.asarray(sparse_graph.arc_length, dtype=float)
        edge_id = np.arange(len(u), dtype=np.int64)

        graph = nx.Graph()
        graph.add_nodes_from(
            (int(i), {"x": float(point[0]), "y": float(point[1])})
            for i, point in enumerate(xy)
        )
        graph.add_edges_from(
            (int(a), int(b), {"length": float(w), "edge_id": int(e)})
            for e, (a, b, w) in enumerate(zip(u, v, length, strict=True))
        )
        nodes = gpd.GeoDataFrame(
            {"network_node": np.arange(len(xy), dtype=np.int64)},
            geometry=gpd.GeoSeries(shapely.points(xy), crs=crs),
            crs=crs,
        )
        lines = shapely.linestrings(np.stack([xy[u], xy[v]], axis=1))
        segments = gpd.GeoDataFrame(
            {"edge_id": edge_id, "u": u, "v": v, "length": length},
            geometry=gpd.GeoSeries(lines, crs=crs),
            crs=crs,
        )
        return cls(crs, graph, nodes, segments, vertex_digits)
    @classmethod
    def from_file(
        cls, path: str, layer: str | None = None, vertex_digits: int = 9
    ) -> "RoadNetwork":
        """Read linework through GeoPandas and build :class:`RoadNetwork`."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"road input does not exist: {source}")
        if source.suffix.lower() in {".parquet", ".pq"}:
            roads = gpd.read_parquet(source)
        else:
            roads = gpd.read_file(source, layer=layer)
        return cls.from_geodataframe(roads, vertex_digits=vertex_digits)

    def snap(
        self,
        points: gpd.GeoDataFrame,
        progress: Callable[[str], None] | None = None,
    ) -> "SnappedPoints":
        """Snap every point continuously to its nearest base-road segment.

        ``snap_distance`` is straight-line QA metadata only.  It is never added to road
        shortest-path distances.  Exact nearest-edge ties are resolved by the smallest
        deterministic ``edge_id``.
        """
        if points.crs is None:
            raise ValueError("point CRS is missing")
        projected = points.to_crs(self.crs) if points.crs != self.crs else points.copy()
        geometries = projected.geometry.to_numpy()

        for position, point in enumerate(geometries):
            if point is None or point.is_empty:
                raise ValueError(f"point row {position} has empty geometry")
            if point.geom_type != "Point":
                raise ValueError(
                    f"point row {position} must have Point geometry, got {point.geom_type}"
                )
            coordinates = np.asarray(point.coords, dtype=float)
            if not np.isfinite(coordinates[:, :2]).all():
                raise ValueError(f"point row {position} has non-finite coordinates")

        lines = self.segments.geometry.to_numpy()
        tree = shapely.STRtree(lines)
        records: list[dict[str, object]] = []

        total_points = len(geometries)
        progress_step = max(1, total_points // 20)
        for position, point in enumerate(geometries):
            if progress is not None and (position == 0 or position % progress_step == 0):
                progress(f"road snap: {position:,}/{total_points:,} points")
            candidates = np.asarray(tree.query_nearest(point, all_matches=True), dtype=np.int64)
            if candidates.size == 0:
                raise RuntimeError(f"no road segment found for point row {position}")

            # ``segments`` is ordered by stable edge_id, therefore the smallest row index is
            # also the deterministic edge-id tie breaker.
            edge_position = int(candidates.min())
            line = lines[edge_position]
            geometric_length = float(line.length)
            measure = float(shapely.line_locate_point(line, point))
            fraction = 0.0 if geometric_length == 0 else measure / geometric_length

            edge = self.segments.iloc[edge_position]
            network_length = float(edge.length)
            offset = float(np.clip(fraction * network_length, 0.0, network_length))
            tolerance = max(1e-10, network_length * 1e-12)
            if offset <= tolerance:
                offset = 0.0
            elif network_length - offset <= tolerance:
                offset = network_length

            snapped_geometry = shapely.line_interpolate_point(line, measure)
            records.append(
                {
                    "source_pos": position,
                    "edge_pos": edge_position,
                    "edge_id": int(edge.edge_id),
                    "offset": offset,
                    "snap_distance": float(point.distance(snapped_geometry)),
                    "snapped_geometry": snapped_geometry,
                }
            )

        if progress is not None:
            progress(f"road snap: {total_points:,}/{total_points:,} points complete")
        return SnappedPoints(pd.DataFrame.from_records(records), self.crs)

    def augment(
        self,
        snapped: "SnappedPoints",
        progress: Callable[[str], None] | None = None,
    ) -> "AugmentedNetwork":
        """Insert all snapped point positions into the road graph exactly once.

        Every base edge containing one or more interior snap positions is replaced by a
        chain of sub-edges whose lengths sum to the original edge length.  Repeated points
        at the same offset share one graph vertex.  ``point_node`` preserves the original
        point-row order and is the bridge between tabular points and network algorithms.
        """
        if snapped.crs != self.crs:
            raise ValueError("snapped-point CRS does not match the road network")
        required = {"source_pos", "edge_pos", "edge_id", "offset"}
        missing = required - set(snapped.frame.columns)
        if missing:
            raise ValueError(f"snapped-point table is missing columns: {sorted(missing)}")

        n_points = len(snapped.frame)
        source_positions = snapped.frame["source_pos"].to_numpy(np.int64, copy=False)
        if sorted(source_positions.tolist()) != list(range(n_points)):
            raise ValueError("snapped source_pos must be a permutation of 0..n-1")

        graph = self.graph.copy()
        node_xy = {
            int(network_node): (float(point.x), float(point.y))
            for network_node, point in zip(
                self.nodes.network_node, self.nodes.geometry, strict=True
            )
        }
        point_node = np.full(n_points, -1, dtype=np.int64)
        next_node = len(node_xy)
        split_rows: list[dict[str, object]] = []

        grouped = snapped.frame.groupby("edge_pos", sort=True)
        touched_edge_count = int(grouped.ngroups)
        progress_step = max(1, touched_edge_count // 20)
        for group_number, (edge_position, group) in enumerate(grouped, start=1):
            if progress is not None and (group_number == 1 or group_number % progress_step == 0):
                progress(
                    f"road augment: {group_number:,}/{touched_edge_count:,} snapped road edges"
                )
            edge_position = int(edge_position)
            if not 0 <= edge_position < len(self.segments):
                raise ValueError(f"invalid snapped edge_pos: {edge_position}")

            edge = self.segments.iloc[edge_position]
            u, v = int(edge.u), int(edge.v)
            network_length = float(edge.length)
            line = edge.geometry
            tolerance = max(1e-10, network_length * 1e-12)

            # Exact duplicate offsets collapse naturally.  Endpoint offsets are represented
            # by their existing endpoint nodes and therefore need no inserted vertex.
            interior_offsets = sorted(
                {
                    float(offset)
                    for offset in group.offset
                    if float(offset) > tolerance
                    and network_length - float(offset) > tolerance
                }
            )
            offset_node: dict[float, int] = {0.0: u, network_length: v}

            for offset in interior_offsets:
                network_node = next_node
                next_node += 1
                geometry = shapely.line_interpolate_point(
                    line,
                    offset / network_length,
                    normalized=True,
                )
                node_xy[network_node] = (float(geometry.x), float(geometry.y))
                graph.add_node(network_node, x=float(geometry.x), y=float(geometry.y))
                offset_node[offset] = network_node

            if interior_offsets:
                if not graph.has_edge(u, v):
                    raise RuntimeError(
                        f"base edge {int(edge.edge_id)} is missing from the road graph"
                    )
                graph.remove_edge(u, v)

            chain = [0.0, *interior_offsets, network_length]
            for start_offset, end_offset in zip(chain[:-1], chain[1:], strict=True):
                start_node = offset_node[start_offset]
                end_node = offset_node[end_offset]
                length = float(end_offset - start_offset)
                if length <= 0:
                    raise RuntimeError("road augmentation produced a non-positive sub-edge")

                geometry = substring(
                    line,
                    start_offset / network_length,
                    end_offset / network_length,
                    normalized=True,
                )
                if geometry.is_empty or geometry.geom_type != "LineString":
                    geometry = LineString([node_xy[start_node], node_xy[end_node]])

                graph.add_edge(
                    start_node,
                    end_node,
                    length=length,
                    parent_edge_id=int(edge.edge_id),
                )
                split_rows.append(
                    {
                        "u": start_node,
                        "v": end_node,
                        "length": length,
                        "parent_edge_id": int(edge.edge_id),
                        "geometry": geometry,
                    }
                )

            # Map every original point row to the inserted/existing node representing its
            # snapped position.  Floating offsets are matched using the same endpoint
            # tolerance used while constructing the chain.
            for _, row in group.iterrows():
                offset = float(row.offset)
                if offset <= tolerance:
                    network_node = u
                elif network_length - offset <= tolerance:
                    network_node = v
                else:
                    if not interior_offsets:
                        raise RuntimeError("interior snapped point has no inserted road node")
                    nearest_offset = min(interior_offsets, key=lambda value: abs(value - offset))
                    network_node = offset_node[nearest_offset]
                point_node[int(row.source_pos)] = network_node

        # Base edges without snapped points were not visited above; copy them into the
        # augmented edge table while leaving the corresponding graph edge untouched.
        split_parent_ids = {int(row["parent_edge_id"]) for row in split_rows}
        for _, edge in self.segments.iterrows():
            edge_id = int(edge.edge_id)
            if edge_id in split_parent_ids:
                continue
            split_rows.append(
                {
                    "u": int(edge.u),
                    "v": int(edge.v),
                    "length": float(edge.length),
                    "parent_edge_id": edge_id,
                    "geometry": edge.geometry,
                }
            )

        if np.any(point_node < 0):
            raise RuntimeError("failed to assign every point to the augmented road graph")

        ordered_nodes = sorted(node_xy)
        nodes = gpd.GeoDataFrame(
            {"network_node": np.asarray(ordered_nodes, dtype=np.int64)},
            geometry=[Point(node_xy[node]) for node in ordered_nodes],
            crs=self.crs,
        )
        edges = gpd.GeoDataFrame(split_rows, geometry="geometry", crs=self.crs)
        edges = edges.sort_values(
            ["parent_edge_id", "u", "v"], kind="stable"
        ).reset_index(drop=True)
        edges["aug_edge_id"] = np.arange(len(edges), dtype=np.int64)

        # Splitting must preserve each base arc's total network length.  This invariant is
        # cheap to check and protects every downstream distance calculation from subtle
        # offset/slicing mistakes.
        split_length = edges.groupby("parent_edge_id", sort=False)["length"].sum()
        base_length = self.segments.set_index("edge_id")["length"]
        for edge_id, expected_length in base_length.items():
            actual_length = float(split_length.loc[int(edge_id)])
            tolerance = max(1e-9, abs(float(expected_length)) * 1e-12)
            if abs(actual_length - float(expected_length)) > tolerance:
                raise RuntimeError(
                    f"road augmentation changed edge {int(edge_id)} length: "
                    f"expected {float(expected_length)}, got {actual_length}"
                )

        if progress is not None:
            progress(
                f"road augment: {touched_edge_count:,}/{touched_edge_count:,} "
                "snapped road edges complete"
            )
        return AugmentedNetwork(self.crs, graph, nodes, edges, point_node)


@dataclass(frozen=True)
class SnappedPoints:
    """Tabular result of continuous point-to-road snapping."""

    frame: pd.DataFrame
    crs: object


@dataclass(frozen=True)
class AugmentedNetwork:
    """Road graph after all analysis-point snap positions have been inserted."""

    crs: object
    graph: nx.Graph
    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    point_node: np.ndarray

    @classmethod
    def from_checkpoint_frames(
        cls,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        point_node: np.ndarray,
    ) -> "AugmentedNetwork":
        """Reconstruct and validate an augmented network from durable checkpoint tables.

        The checkpoint stores portable GeoParquet tables rather than a pickled NetworkX
        object.  On restore we rebuild the lightweight graph container and verify the
        invariants required by the sparse shortest-path representation.  This makes a
        damaged or incompatible checkpoint fail closed so the pipeline can rebuild it.
        """
        required_nodes = {"network_node", "geometry"}
        required_edges = {"u", "v", "length", "parent_edge_id", "aug_edge_id", "geometry"}
        missing_nodes = required_nodes - set(nodes.columns)
        missing_edges = required_edges - set(edges.columns)
        if missing_nodes:
            raise ValueError(
                f"augmented-road node checkpoint is missing columns: {sorted(missing_nodes)}"
            )
        if missing_edges:
            raise ValueError(
                f"augmented-road edge checkpoint is missing columns: {sorted(missing_edges)}"
            )
        if nodes.crs is None or edges.crs is None:
            raise ValueError("augmented-road checkpoint CRS is missing")
        if nodes.crs != edges.crs:
            raise ValueError("augmented-road checkpoint node/edge CRS mismatch")

        restored_nodes = nodes.sort_values("network_node", kind="stable").reset_index(drop=True)
        node_ids = restored_nodes["network_node"].to_numpy(np.int64, copy=False)
        expected_ids = np.arange(len(restored_nodes), dtype=np.int64)
        if not np.array_equal(node_ids, expected_ids):
            raise ValueError("augmented-road checkpoint node IDs are not contiguous from zero")
        if restored_nodes.geometry.is_empty.any() or restored_nodes.geometry.isna().any():
            raise ValueError("augmented-road checkpoint contains empty node geometry")
        if not (restored_nodes.geometry.geom_type == "Point").all():
            raise ValueError("augmented-road checkpoint nodes must be Point geometries")

        restored_edges = edges.sort_values("aug_edge_id", kind="stable").reset_index(drop=True)
        aug_edge_ids = restored_edges["aug_edge_id"].to_numpy(np.int64, copy=False)
        if not np.array_equal(aug_edge_ids, np.arange(len(restored_edges), dtype=np.int64)):
            raise ValueError("augmented-road checkpoint edge IDs are not contiguous from zero")
        u = restored_edges["u"].to_numpy(np.int64, copy=False)
        v = restored_edges["v"].to_numpy(np.int64, copy=False)
        length = restored_edges["length"].to_numpy(float, copy=False)
        if len(restored_edges) == 0:
            raise ValueError("augmented-road checkpoint contains no edges")
        if (
            u.min() < 0
            or v.min() < 0
            or u.max() >= len(restored_nodes)
            or v.max() >= len(restored_nodes)
        ):
            raise ValueError("augmented-road checkpoint edge references an unknown node")
        if not np.isfinite(length).all() or (length <= 0).any():
            raise ValueError("augmented-road checkpoint contains invalid edge length")
        if restored_edges.geometry.is_empty.any() or restored_edges.geometry.isna().any():
            raise ValueError("augmented-road checkpoint contains empty edge geometry")

        point_node = np.asarray(point_node, dtype=np.int64)
        if point_node.ndim != 1:
            raise ValueError("augmented-road point-node checkpoint must be one-dimensional")
        if point_node.size and (point_node.min() < 0 or point_node.max() >= len(restored_nodes)):
            raise ValueError("augmented-road point-node checkpoint references an unknown node")

        graph = nx.Graph()
        xy = shapely.get_coordinates(restored_nodes.geometry.to_numpy())
        if len(xy) != len(restored_nodes):
            raise ValueError("augmented-road checkpoint node geometry is malformed")
        graph.add_nodes_from(
            (int(node), {"x": float(coord[0]), "y": float(coord[1])})
            for node, coord in zip(node_ids, xy, strict=True)
        )
        graph.add_edges_from(
            (
                int(row.u),
                int(row.v),
                {
                    "length": float(row.length),
                    "parent_edge_id": int(row.parent_edge_id),
                    "aug_edge_id": int(row.aug_edge_id),
                },
            )
            for row in restored_edges.itertuples(index=False)
        )
        if graph.number_of_edges() != len(restored_edges):
            raise ValueError("augmented-road checkpoint contains duplicate graph edges")

        return cls(nodes.crs, graph, restored_nodes, restored_edges, point_node.copy())

    @cached_property
    def sparse_adjacency(self):
        """Symmetric CSR adjacency reused by all shortest-path queries."""
        n = len(self.nodes)
        u = self.edges["u"].to_numpy(np.int64, copy=False)
        v = self.edges["v"].to_numpy(np.int64, copy=False)
        w = self.edges["length"].to_numpy(float, copy=False)
        matrix = coo_matrix(
            (np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)
        ).tocsr()
        matrix.sort_indices()
        return matrix

    @cached_property
    def sparse_component(self) -> np.ndarray:
        _, component = connected_components(self.sparse_adjacency, directed=False)
        return np.asarray(component, dtype=np.int64)

    def component_id(self) -> dict[Hashable, int]:
        """Return deterministic connected-component IDs for all graph nodes."""
        return {int(node): int(comp) for node, comp in enumerate(self.sparse_component)}

    def distance(self, source: int, target: int) -> float:
        """Shortest road distance from the cached SciPy graph."""
        source, target = int(source), int(target)
        n = self.sparse_adjacency.shape[0]
        if not (0 <= source < n and 0 <= target < n):
            raise ValueError(f"unknown network node in distance query: {source}, {target}")
        if self.sparse_component[source] != self.sparse_component[target]:
            return float("inf")
        return float(
            np.asarray(
                dijkstra(self.sparse_adjacency, directed=True, indices=source), dtype=float
            )[target]
        )

    def distances_from(self, source: int, cutoff: float | None = None) -> dict[int, float]:
        """Single-source shortest paths from the cached SciPy graph."""
        source = int(source)
        n = self.sparse_adjacency.shape[0]
        if not 0 <= source < n:
            raise ValueError(f"unknown network source node: {source}")
        if cutoff is not None and (not np.isfinite(cutoff) or cutoff < 0):
            raise ValueError("cutoff must be finite and non-negative")
        limit = np.inf if cutoff is None else float(cutoff)
        values = np.asarray(
            dijkstra(self.sparse_adjacency, directed=True, indices=source, limit=limit),
            dtype=float,
        )
        reached = np.flatnonzero(np.isfinite(values))
        return {int(node): float(values[node]) for node in reached}

    def distances_to_targets(
        self, source: int, targets: np.ndarray | list[int] | tuple[int, ...]
    ) -> np.ndarray:
        """Shortest-path distances from one source to requested target nodes only.

        SciPy still computes one bounded-memory single-source distance vector internally,
        but this method immediately extracts the requested target entries and discards the
        full vector.  It therefore avoids the large Python dictionaries formerly cached by
        spatial-network instantiation.
        """
        source = int(source)
        target_array = np.asarray(targets, dtype=np.int64)
        if target_array.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        n = self.sparse_adjacency.shape[0]
        if not 0 <= source < n:
            raise ValueError(f"unknown network source node: {source}")
        if target_array.size == 0:
            return np.empty(0, dtype=float)
        if target_array.min() < 0 or target_array.max() >= n:
            raise ValueError("target list contains an unknown network node")

        out = np.full(target_array.shape, np.inf, dtype=float)
        same_component = self.sparse_component[target_array] == self.sparse_component[source]
        if not bool(same_component.any()):
            return out
        values = np.asarray(
            dijkstra(self.sparse_adjacency, directed=True, indices=source),
            dtype=float,
        )
        out[same_component] = values[target_array[same_component]]
        return out


    def shortest_paths_to_targets(
        self, source: int, targets: np.ndarray | list[int] | tuple[int, ...]
    ) -> tuple[np.ndarray, list[tuple[int, ...] | None]]:
        """Return exact shortest distances and road-node paths to requested targets.

        This is the geometry-preserving counterpart of :meth:`distances_to_targets`.
        One SciPy Dijkstra run is performed for the source with a predecessor vector,
        then only the requested target paths are reconstructed.  The predecessor vector
        is discarded on return, so memory remains bounded by one single-source solve.
        """
        source = int(source)
        target_array = np.asarray(targets, dtype=np.int64)
        if target_array.ndim != 1:
            raise ValueError("targets must be one-dimensional")
        n = self.sparse_adjacency.shape[0]
        if not 0 <= source < n:
            raise ValueError(f"unknown network source node: {source}")
        if target_array.size == 0:
            return np.empty(0, dtype=float), []
        if target_array.min() < 0 or target_array.max() >= n:
            raise ValueError("target list contains an unknown network node")

        out = np.full(target_array.shape, np.inf, dtype=float)
        paths: list[tuple[int, ...] | None] = [None] * len(target_array)
        same_component = self.sparse_component[target_array] == self.sparse_component[source]
        if not bool(same_component.any()):
            return out, paths

        values, predecessors = dijkstra(
            self.sparse_adjacency,
            directed=True,
            indices=source,
            return_predecessors=True,
        )
        values = np.asarray(values, dtype=float)
        predecessors = np.asarray(predecessors, dtype=np.int64)
        out[same_component] = values[target_array[same_component]]

        for position in np.flatnonzero(same_component):
            target = int(target_array[position])
            if target == source:
                paths[int(position)] = (source,)
                continue
            if not np.isfinite(out[position]):
                continue
            reverse_path = [target]
            cursor = target
            # SciPy uses -9999 for an unreachable/no-predecessor vertex.  A finite
            # distance in the same connected component should always reach source.
            for _ in range(n):
                predecessor = int(predecessors[cursor])
                if predecessor < 0:
                    raise RuntimeError(
                        f"could not reconstruct shortest road path {source}->{target}"
                    )
                reverse_path.append(predecessor)
                cursor = predecessor
                if cursor == source:
                    break
            else:
                raise RuntimeError(
                    f"shortest road path {source}->{target} exceeded graph size"
                )
            paths[int(position)] = tuple(reversed(reverse_path))
        return out, paths

    @cached_property
    def _edge_geometry_lookup(self) -> dict[tuple[int, int], tuple[LineString, bool]]:
        """Map either orientation of an augmented graph edge to its stored line geometry."""
        lookup: dict[tuple[int, int], tuple[LineString, bool]] = {}
        for row in self.edges.itertuples(index=False):
            u, v = int(row.u), int(row.v)
            geometry = row.geometry
            lookup[(u, v)] = (geometry, False)
            lookup[(v, u)] = (geometry, True)
        return lookup

    def path_geometry(self, path_nodes: tuple[int, ...] | list[int]) -> LineString:
        """Convert a shortest-path road-node sequence to one oriented LineString.

        The stored augmented edge geometry is used, not straight center-to-center chords.
        A zero-distance path (source and target are the same inserted road node) is encoded
        as a two-coordinate zero-length LineString so it can still be stored in a line layer.
        """
        nodes = tuple(int(node) for node in path_nodes)
        if not nodes:
            raise ValueError("path_nodes must not be empty")
        if len(nodes) == 1:
            point = self.nodes.loc[
                self.nodes["network_node"].astype(int) == nodes[0], "geometry"
            ]
            if len(point) != 1:
                raise RuntimeError(f"unknown network node in road path: {nodes[0]}")
            xy = tuple(point.iloc[0].coords[0][:2])
            return LineString([xy, xy])

        coordinates: list[tuple[float, float]] = []
        for u, v in zip(nodes[:-1], nodes[1:], strict=True):
            try:
                geometry, reverse = self._edge_geometry_lookup[(u, v)]
            except KeyError as exc:
                raise RuntimeError(f"road path references missing edge {u}<->{v}") from exc
            edge_coordinates = [tuple(value[:2]) for value in geometry.coords]
            if reverse:
                edge_coordinates.reverse()
            if coordinates and edge_coordinates:
                if coordinates[-1] == edge_coordinates[0]:
                    coordinates.extend(edge_coordinates[1:])
                else:
                    coordinates.extend(edge_coordinates)
            else:
                coordinates.extend(edge_coordinates)
        if len(coordinates) < 2:
            raise RuntimeError("road path geometry contains fewer than two coordinates")
        return LineString(coordinates)
