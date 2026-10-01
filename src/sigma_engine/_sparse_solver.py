"""Shared SciPy shortest-path and network-median backend.

Shortest paths use Dijkstra's algorithm (Dijkstra, 1959) through SciPy
(Virtanen et al., 2020, https://doi.org/10.1038/s41592-019-0686-2).  The
vertex 1-median objective follows classical network-location theory: Hakimi
(1964), https://doi.org/10.1287/opre.12.3.450.  For vertex demand a median
optimum can be chosen at a network vertex; SIGMA evaluates that minisum
objective in bounded SciPy distance blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .network import AugmentedNetwork

_MAX_MATRIX_VALUES = 8_000_000


@dataclass(frozen=True)
class SparseRoadSolver:
    adjacency: csr_matrix
    component: np.ndarray

    @classmethod
    def from_augmented(cls, network: AugmentedNetwork) -> "SparseRoadSolver":
        n = len(network.nodes)
        node_ids = network.nodes["network_node"].to_numpy(np.int64, copy=False)
        if not np.array_equal(node_ids, np.arange(n, dtype=np.int64)):
            raise ValueError("optimized road solver requires contiguous network_node IDs")
        graph = network.sparse_adjacency
        component = network.sparse_component
        return cls(graph, component)

    def distance(self, source: int, target: int) -> float:
        source, target = int(source), int(target)
        if not (0 <= source < self.adjacency.shape[0] and 0 <= target < self.adjacency.shape[0]):
            raise ValueError(f"unknown network node in distance query: {source}, {target}")
        if self.component[source] != self.component[target]:
            return float("inf")
        value = dijkstra(
            self.adjacency,
            directed=True,
            indices=source,
            min_only=False,
            limit=np.inf,
        )[target]
        return float(value)

    def distances_from(self, source: int, cutoff: float | None = None) -> dict[int, float]:
        source = int(source)
        if not 0 <= source < self.adjacency.shape[0]:
            raise ValueError(f"unknown network source node: {source}")
        limit = np.inf if cutoff is None else float(cutoff)
        values = np.asarray(
            dijkstra(self.adjacency, directed=True, indices=source, limit=limit), dtype=float
        )
        idx = np.flatnonzero(np.isfinite(values))
        return {int(i): float(values[i]) for i in idx}

    def one_median(self, demand_nodes: np.ndarray) -> tuple[int, float]:
        """Exact weighted vertex 1-median for repeated unit-weight demand nodes."""
        demand = np.asarray(demand_nodes, dtype=np.int64)
        if demand.ndim != 1 or len(demand) == 0:
            raise ValueError("network 1-median requires a non-empty one-dimensional demand array")
        if demand.min() < 0 or demand.max() >= self.adjacency.shape[0]:
            raise ValueError("network 1-median received an unknown demand node")
        comp = self.component[demand[0]]
        if np.any(self.component[demand] != comp):
            raise ValueError("one cluster spans disconnected road components")

        unique, weights = np.unique(demand, return_counts=True)
        candidates = np.flatnonzero(self.component == comp).astype(np.int64)
        local_of = np.full(self.adjacency.shape[0], -1, dtype=np.int64)
        local_of[candidates] = np.arange(len(candidates), dtype=np.int64)
        local_demand = local_of[unique]
        sub = self.adjacency[candidates][:, candidates]
        scores = np.zeros(len(candidates), dtype=float)

        batch = max(1, _MAX_MATRIX_VALUES // max(1, len(candidates)))
        for start in range(0, len(unique), batch):
            stop = min(len(unique), start + batch)
            dist = np.asarray(
                dijkstra(sub, directed=True, indices=local_demand[start:stop]), dtype=float
            )
            scores += weights[start:stop].astype(float) @ dist

        best = float(scores.min())
        tolerance = max(1e-9, abs(best) * 1e-12)
        tied = candidates[scores <= best + tolerance]
        best_node = int(tied.min())
        return best_node, float(scores[local_of[best_node]])

    def point_distances_from_centers(
        self, point_nodes: np.ndarray, center_for_point: np.ndarray
    ) -> np.ndarray:
        """Road distance from each point node to its assigned center, batched by center."""
        points = np.asarray(point_nodes, dtype=np.int64)
        centers = np.asarray(center_for_point, dtype=np.int64)
        if points.shape != centers.shape:
            raise ValueError("point and center node arrays must align")
        out = np.empty(len(points), dtype=float)
        for center in np.unique(centers):
            mask = centers == center
            values = np.asarray(
                dijkstra(self.adjacency, directed=True, indices=int(center)), dtype=float
            )
            out[mask] = values[points[mask]]
        return out
