"""Spatial instantiation of the MWAS IO DAG and directed centrality on ``X``.

The economic DAG supplied here is the acyclic subgraph selected from the weighted
IO network by SIGMA's MWAS stage.  This module does not alter its sector-level
direction or retained IO edge weight; it instantiates each surviving economic
edge only where the corresponding spatial partitions overlap, then attaches
road-network center distance as a separate attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from .network import AugmentedNetwork


_DIAGNOSTIC_COLUMNS = ["type1", "cluster1", "type2", "cluster2", "status"]


def make_node_id(type_value: str, cluster: int) -> str:
    """Encode a stable, reversible-enough textual key for ``[type, cluster]``.

    URL quoting prevents sector labels containing spaces, slashes, or the ``::`` delimiter
    from making audit files ambiguous.  ``type`` and ``cluster`` remain separate columns in
    every artifact, so consumers never need to parse this string to recover them.
    """
    return f"type={quote(str(type_value), safe='')}::cluster={int(cluster)}"


@dataclass(frozen=True)
class CentralityResult:
    """Directed weighted eigenvector-centrality result plus interpretation metadata."""

    values: dict[str, float]
    direction: str
    degenerate_dag: bool
    note: str | None


def _positive_area_overlaps(left_geometry, right: gpd.GeoDataFrame) -> list[int]:
    """Return right-row positions having positive-area intersection with ``left``."""
    candidates = np.asarray(
        right.sindex.query(left_geometry, predicate="intersects"),
        dtype=np.int64,
    )
    overlaps: list[int] = []
    for position in candidates:
        intersection = left_geometry.intersection(right.geometry.iloc[int(position)])
        if not intersection.is_empty and float(intersection.area) > 0:
            overlaps.append(int(position))
    return overlaps


def _validate_spatial_inputs(
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
    io_dag: nx.DiGraph,
) -> None:
    """Check join keys and graph invariants before expensive spatial work."""
    if not nx.is_directed_acyclic_graph(io_dag):
        raise ValueError("spatial instantiation requires an acyclic IO graph")

    center_required = {"type", "cluster", "center_network_node", "geometry"}
    partition_required = {"type", "cluster", "geometry"}
    missing_centers = center_required - set(centers.columns)
    missing_partitions = partition_required - set(partitions.columns)
    if missing_centers:
        raise ValueError(f"center table is missing columns: {sorted(missing_centers)}")
    if missing_partitions:
        raise ValueError(f"partition table is missing columns: {sorted(missing_partitions)}")
    if centers.crs is None or partitions.crs is None:
        raise ValueError("centers and partitions must both have a CRS")
    if centers.crs != partitions.crs:
        raise ValueError("centers and partitions must use the same CRS")

    for position, geometry in enumerate(centers.geometry):
        if geometry is None or geometry.is_empty or geometry.geom_type != "Point":
            raise ValueError(f"center row {position} must contain one non-empty Point")
        coordinates = np.asarray(geometry.coords, dtype=float)
        if not np.isfinite(coordinates[:, :2]).all():
            raise ValueError(f"center row {position} contains non-finite coordinates")

    for position, geometry in enumerate(partitions.geometry):
        if geometry is None or geometry.is_empty:
            raise ValueError(f"partition row {position} has empty geometry")
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"partition row {position} must be polygonal")
        if not geometry.is_valid:
            raise ValueError(
                f"partition row {position} is invalid: {shapely.is_valid_reason(geometry)}"
            )
        if not np.isfinite(float(geometry.area)) or float(geometry.area) <= 0:
            raise ValueError(f"partition row {position} must have positive finite area")

    center_keys = list(
        zip(
            centers["type"].astype(str),
            centers["cluster"].astype(int),
            strict=True,
        )
    )
    partition_keys = list(
        zip(partitions["type"].astype(str), partitions["cluster"].astype(int), strict=True)
    )
    if len(center_keys) != len(set(center_keys)):
        raise ValueError("center [type, cluster] keys must be unique")
    if len(partition_keys) != len(set(partition_keys)):
        raise ValueError("partition [type, cluster] keys must be unique")

    unknown_partitions = sorted(set(partition_keys) - set(center_keys))
    if unknown_partitions:
        raise ValueError(
            "partition rows have no matching network center: "
            f"{unknown_partitions[:10]}"
        )


def instantiate_network(
    centers: gpd.GeoDataFrame,
    partitions: gpd.GeoDataFrame,
    io_dag: nx.DiGraph,
    road: AugmentedNetwork,
    progress: Callable[[str], None] | None = None,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Create cluster-level directed network ``X`` from spatially overlapping partitions.

    Every center becomes a node before edges are considered, so isolated clusters are
    retained.  For each IO-DAG type edge, only positive-area polygon overlaps create a
    cluster edge.  The road distance is measured between the two network 1-medians.
    """
    _validate_spatial_inputs(centers, partitions, io_dag)

    graph = nx.DiGraph()
    center_index: dict[tuple[str, int], pd.Series] = {}
    for _, row in centers.iterrows():
        key = (str(row["type"]), int(row["cluster"]))
        node_id = make_node_id(*key)
        center_index[key] = row
        graph.add_node(
            node_id,
            type=key[0],
            cluster=key[1],
            center_network_node=int(row["center_network_node"]),
            center_x=float(row.geometry.x),
            center_y=float(row.geometry.y),
        )

    diagnostics: list[dict[str, object]] = []
    distance_cache: dict[int, dict[int, float]] = {}
    partitions_by_type = {
        str(type_value): group.reset_index(drop=True)
        for type_value, group in partitions.groupby("type", sort=False)
    }

    active_edges = [
        (type1, type2, edge_data)
        for type1, type2, edge_data in io_dag.edges(data=True)
        if str(type1) in partitions_by_type and str(type2) in partitions_by_type
    ]
    total_active_edges = len(active_edges)
    edge_step = max(1, total_active_edges // 20)

    for edge_number, (type1, type2, edge_data) in enumerate(active_edges, start=1):
        type1, type2 = str(type1), str(type2)
        if progress is not None and (
            edge_number == 1 or edge_number % edge_step == 0 or edge_number == total_active_edges
        ):
            progress(
                f"network X: IO edge {edge_number:,}/{total_active_edges:,} "
                f"{type1}->{type2}; current X edges={graph.number_of_edges():,}"
            )

        dag_weight = float(edge_data["weight"])
        if not np.isfinite(dag_weight) or dag_weight <= 0:
            raise ValueError(
                f"IO-DAG edge {type1!r}->{type2!r} has non-positive/non-finite weight"
            )

        source_partitions = partitions_by_type[type1]
        target_partitions = partitions_by_type[type2]

        for _, source_partition in source_partitions.iterrows():
            target_positions = _positive_area_overlaps(
                source_partition.geometry,
                target_partitions,
            )
            for target_position in target_positions:
                target_partition = target_partitions.iloc[target_position]
                cluster1 = int(source_partition["cluster"])
                cluster2 = int(target_partition["cluster"])

                source_center = center_index[(type1, cluster1)]
                target_center = center_index[(type2, cluster2)]
                source_network_node = int(source_center["center_network_node"])
                target_network_node = int(target_center["center_network_node"])

                if source_network_node not in distance_cache:
                    distance_cache[source_network_node] = road.distances_from(source_network_node)
                road_distance = distance_cache[source_network_node].get(
                    target_network_node,
                    float("inf"),
                )

                if not np.isfinite(road_distance):
                    diagnostics.append(
                        {
                            "type1": type1,
                            "cluster1": cluster1,
                            "type2": type2,
                            "cluster2": cluster2,
                            "status": "disconnected_centers",
                        }
                    )
                    continue
                if road_distance < 0:
                    raise RuntimeError("shortest-path engine returned a negative road distance")

                # The requested SIGMA edge definition is multiplicative.  This makes longer
                # road distance increase edge weight (rather than act as a distance penalty);
                # the implementation intentionally follows that specification literally.
                combined_weight = dag_weight * float(road_distance)
                graph.add_edge(
                    make_node_id(type1, cluster1),
                    make_node_id(type2, cluster2),
                    dag_weight=dag_weight,
                    road_distance=float(road_distance),
                    weight=combined_weight,
                )

    # Because every cluster edge projects onto one IO-DAG edge, a directed cycle in X would
    # imply a directed cycle in the type-level DAG.  Treat violation as an implementation
    # error rather than letting centrality operate on a graph with unexpected semantics.
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("spatial instantiation produced a cycle from an acyclic IO graph")

    if progress is not None:
        progress(
            f"network X: complete ({graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges, {len(diagnostics):,} disconnected overlaps)"
        )
    return graph, pd.DataFrame(diagnostics, columns=_DIAGNOSTIC_COLUMNS)


def _positive_weight_support(graph: nx.DiGraph) -> nx.DiGraph:
    """Return the adjacency support that actually contributes to weighted centrality.

    A spatial edge can have weight zero when two network medians occupy the same road node.
    Such an edge is present for auditability but contributes a zero adjacency entry.  It must
    therefore not change which nodes form the kernel of a DAG adjacency matrix.
    """
    support = nx.DiGraph()
    support.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                f"centrality edge {source!r}->{target!r} has invalid weight {weight!r}"
            )
        if weight > 0:
            support.add_edge(source, target, **data)
    return support


def directed_eigenvector_centrality(
    graph: nx.DiGraph,
    direction: str = "incoming",
) -> CentralityResult:
    """Compute weighted directed eigenvector centrality with an explicit direction.

    ``incoming`` uses the conventional left-eigenvector interpretation: a node is important
    when important predecessors point to it.  ``outgoing`` applies the same definition to
    the reversed graph.

    For the SIGMA workflow, ``X`` is a DAG.  Its adjacency matrix is nilpotent and therefore
    has spectral radius zero, so there is no unique positive Perron vector.  In that exact
    case SIGMA returns one deterministic non-negative unit-norm zero-eigenvalue vector:
    equal mass on positive-weight terminal nodes for ``incoming`` (initial nodes for
    ``outgoing``), and zero elsewhere.  Metadata marks this result as degenerate.
    """
    if direction not in {"incoming", "outgoing"}:
        raise ValueError("centrality direction must be 'incoming' or 'outgoing'")
    if graph.number_of_nodes() == 0:
        return CentralityResult({}, direction, False, None)

    support = _positive_weight_support(graph)
    work = support if direction == "incoming" else support.reverse(copy=False)

    if nx.is_directed_acyclic_graph(work):
        terminal_nodes = sorted(
            [node for node in work.nodes if work.out_degree(node) == 0],
            key=str,
        )
        # A finite DAG always has a terminal node; the fallback is defensive for custom
        # graph-like objects and keeps the normalization well-defined.
        if not terminal_nodes:
            terminal_nodes = sorted(work.nodes, key=str)

        scale = 1.0 / np.sqrt(len(terminal_nodes))
        terminal_set = set(terminal_nodes)
        values = {
            str(node): (scale if node in terminal_set else 0.0)
            for node in work.nodes
        }
        note = (
            "X is a DAG, so its positive-weight adjacency matrix is nilpotent and has "
            "spectral radius zero. Eigenvector centrality is non-unique; SIGMA uses equal "
            "unit-norm mass on terminal nodes (after applying the selected direction) and "
            "zero on non-terminal nodes. Zero-weight edges do not alter this support."
        )
        return CentralityResult(values, direction, True, note)

    # The branch below is mainly useful if this function is reused outside the strict SIGMA
    # DAG pipeline.  NetworkX power iteration is preferred; dense eigendecomposition is a
    # deterministic numerical fallback for smaller graphs that fail to converge.
    try:
        values = nx.eigenvector_centrality(
            work,
            weight="weight",
            max_iter=5_000,
            tol=1e-12,
        )
    except nx.PowerIterationFailedConvergence:
        node_order = list(work.nodes)
        matrix = nx.to_numpy_array(
            work,
            nodelist=node_order,
            weight="weight",
            dtype=float,
        )
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        index = int(np.argmax(np.abs(eigenvalues)))
        vector = np.abs(np.real(eigenvectors[:, index]))
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(norm):
            raise RuntimeError("directed eigenvector centrality is numerically undefined")
        values = {
            str(node): float(value / norm)
            for node, value in zip(node_order, vector, strict=True)
        }

    result = {str(node): float(value) for node, value in values.items()}
    if not np.isfinite(np.fromiter(result.values(), dtype=float)).all():
        raise RuntimeError("directed eigenvector centrality produced non-finite values")
    return CentralityResult(result, direction, False, None)
