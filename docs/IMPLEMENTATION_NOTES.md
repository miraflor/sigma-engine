# Revised workflow implementation notes

## Baseline inspected

The overlay was prepared against `miraflor/sigma-engine` `main` at commit:

```text
b02185c2f2e821348e1b03bbcf2cb3640e0588e0
2026-10-01 16:58:42 UTC
Add durable augmented-road checkpoints
```

That baseline identifies itself as `0.2.7` and already contains working internal implementations for:

- sparse road-network HDBSCAN;
- continuous road snapping and augmented road graphs;
- exact network 1-medians;
- network-Voronoi partitions dissolved to clusters;
- fast deterministic MWAS and optional exact MILP MWAS;
- positive-area partition overlap testing;
- targeted median-to-median road distances;
- directed weighted centrality.

Those primitives are deliberately retained rather than rewritten.

## Baseline mismatches corrected

### 1. Step 5 used one matrix for both selection and downstream weight

The baseline MWAS graph kept transaction values as `weight`, and `sigma_io_dag.csv` exposed only that value. The revised workflow requires two matrices with separate roles.

`io_workflow.py` now enforces:

```text
Z -> MWAS -> surviving edge set
A -> reweight only those surviving edges
```

The downstream graph's `weight` is therefore `A_ab`, while `transaction_value=z_ab` is retained for audit.

### 2. Step 7 used inverse distance

The baseline score divided cluster centrality by point-center distance (with a finite zero-distance floor). The revised score is now:

```text
R_sc = Q90(d_sc*)
q_sci = min(d_sci / max(R_sc, 1), 1)
S_sci = C_sc * (1 - lambda*q_sci)
```

with `lambda=0.15` by default.

### 3. Hidden augmented-road checkpoint was required for efficient reuse

The baseline's latest commit added a durable augmented-road checkpoint. The revised restart boundary is instead the public Step-3/4 artifacts. `from-partitions` ignores old augmented node IDs and reconstructs a center-only augmented graph from saved median geometries.

### 4. Fail-soft spatial substitutions conflicted with the clean invariants

The baseline could fall back from an exact network 1-median to a Euclidean centroid and from network Voronoi to Euclidean Voronoi. The revised full run does neither. Required-object failures are explicit errors.

### 5. Artifact terminology differed from the revised specification

The revised public schemas use:

```text
source_type / target_type
transaction_value
technical_coefficient
road_distance
edge_weight
cluster_centrality
distance_to_cluster_median
cluster_distance_p90
normalized_cluster_distance
distance_tempering
sigma_score
```

The restart reader accepts the old `network_distance_to_center` name only as an input compatibility alias. It also requires the legacy audit field to say `point_center_distance_method=network_shortest_path`; an old Euclidean fallback is not silently reused in the revised score.

## Step mapping

| Spec step | Implementation |
|---|---|
| 1 | `prepare_point_input`, `prepare_sparse_context`, retained road augmentation |
| 2 | `cluster_by_type`, `retained_points` |
| 3 | `cluster_centers`, `_point_median_distances` |
| 4 | `all_surface_partitions` with network-only refinement; no Euclidean fallback |
| 5 | `io_workflow.build_io_dag` |
| 6 | existing `spatial_graph.instantiate_network` supplied with the A-reweighted DAG |
| 7 | `scoring.tempered_point_scores` |

## Important deliberate non-change

The specification says "selected directed weighted centrality" but does not define a replacement centrality. The overlay therefore preserves the current directed weighted eigenvector implementation. Because `X` is a DAG, that implementation explicitly records the degenerate nilpotent-adjacency case in run metadata. Changing centrality would be a separate methodological decision, not an implementation correction to this specification.

## Restart contract

`from-partitions` requires:

```text
sigma_partitions.parquet
sigma_network_centers.parquet
sigma_points_with_center_distance.parquet
roads
transactions Z (bundled by default)
technical coefficients A (required)
```

If centers/point-distance paths are omitted, they are resolved beside the partition file.

The command executes no HDBSCAN, no 1-median optimization, and no Voronoi computation.
