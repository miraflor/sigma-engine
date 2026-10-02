# SIGMA Engine

This is the **revised clean workflow reset (`v0.1.0`)**. It implements SIGMA as a direct seven-step dataflow from classified IO points to a cluster-level spatial-economic graph and a bounded point-level score.

The implementation deliberately keeps the existing road-network HDBSCAN, exact network 1-median, network-Voronoi, MWAS, and directed weighted centrality primitives. The reset changes the orchestration and contracts so they match the revised workflow exactly.

## Workflow

```text
classified points
      |
      v
network HDBSCAN by type
      |
      +----------------------------+
      |                            |
      v                            v
exact network 1-medians       network Voronoi by point
+ point->own-median distance  + dissolve by [type, cluster]
      |                            |
      +-------------+--------------+
                    |
transactions Z -> MWAS (fast default) -> surviving sector edges
technical coefficients A -----------> reweight surviving edges
                    |
                    v
spatial-economic graph X
  edge exists iff:
    1. sector edge survived MWAS
    2. cluster partitions overlap with positive area
    3. cluster medians are road-connected
  weight = technical_coefficient * median_road_distance
                    |
                    v
cluster centrality
                    |
                    v
point score = cluster_centrality * (1 - lambda * bounded_cluster_distance)
```

The key Step-5 rule is strict: **transactions select edges; technical coefficients weight the surviving DAG**. The package now ships the PSA 2018 IO80 and IO16 intermediate transaction blocks together with their aligned gross-output vectors, so normal runs derive `A_ij = Z_ij / x_j` internally. SIGMA never substitutes the intermediate-input column sum for gross output.

## Primary artifacts

| Step | Artifact | Contract |
|---|---|---|
| 2 | `sigma_clustered_points.parquet` | clustered classified points; noise has `cluster=-1` |
| 3 | `sigma_network_centers.parquet` | exactly one exact network 1-median per retained `[type, cluster]` |
| 3 | `sigma_points_with_center_distance.parquet` | one road distance from each retained point to its own median |
| 4 | `sigma_partitions.parquet` | exactly one Polygon/MultiPolygon per retained `[type, cluster]` |
| 5 | `sigma_io_dag.csv` | MWAS edge selected with transaction value, then reweighted with technical coefficient |
| 6 | `sigma_X_nodes.csv` | one node per retained cluster, with centrality |
| 6 | `sigma_X_edges.csv` | qualifying directed cluster pairs with `A`, road distance, and `A*d` |
| 6 | `sigma_disconnected_overlap_pairs.csv` | positive-area overlaps whose medians are road-disconnected |
| 7 | `sigma_points_centrality.parquet` | final bounded point-level SIGMA score |

The public Step-2/3/4 artifacts are also the durable restart boundaries. The reset does not hide a second competing semantic state inside checkpoint-only files.

## Built-in PSA economic inputs

Normal runs require **no external IO file**. `--classification io80` (default) uses the bundled PSA 2018 80-industry transaction block and gross-output vector; `--classification io16` uses the corresponding 16-industry resources. Rows are suppliers and columns are users.

Step 5 therefore resolves internally as:

```text
bundled PSA transactions Z -> MWAS -> surviving sector edges
bundled PSA total output x -> A_ij = Z_ij / x_j -> downstream edge weights
```

The bundled resources are preprocessed from the PSA 2018 benchmark transaction tables: presentation-only headings/final-demand columns are omitted, sector order is normalized to canonical SIGMA codes, and the published `Total Output` values are retained separately as compact `sector,total_output` files. Resource hashes are verified at load time.

`--transactions` remains an explicit custom override. If it is a full PSA-style workbook and `--technical-coefficients` is omitted, SIGMA can read its `Total Output` column and derive `A` from that override. `--technical-coefficients` remains an explicit A override. Neither option is needed for the built-in IO80/IO16 workflow.

## Full run

```powershell
sigma-engine run `
  --points "C:\path\classified_points.parquet" `
  --roads "C:\path\roads.gpkg" `
  --boundary "C:\path\boundary.gpkg" `
  --output-dir "C:\path\sigma_output"
```

Fast MWAS is the default. Exact MWAS is opt-in:

```powershell
--mwas-method exact
```

The default point-distance tempering is:

```text
--distance-tempering 0.15
```

For each cluster, `R` is the 90th percentile of point-to-median road distance,

```text
q = min(d / max(R, 1), 1)
score = cluster_centrality * (1 - 0.15*q)
```

so the default score lies between `0.85*C` and `C`.

## Continue immediately from existing partitions

This command is specifically for an output directory where Steps 2--4 were already computed:

```powershell
sigma-engine from-partitions `
  --partitions "C:\path\old_output\sigma_partitions.parquet" `
  --roads "C:\path\roads.gpkg" `
  --output-dir "C:\path\revised_output"
```

By default it auto-detects these sibling files beside `sigma_partitions.parquet`:

```text
sigma_network_centers.parquet
sigma_points_with_center_distance.parquet
```

You can override them with `--centers` and `--points-with-center-distance`.

For compatibility with the current pre-reset repository output, the restart path accepts the old field `network_distance_to_center` and maps it to the revised `distance_to_cluster_median` field. When that legacy alias is used, the artifact must also identify `point_center_distance_method=network_shortest_path`; legacy Euclidean fallback distances are rejected because Step 7 requires road-network distance.

### Why the restart does not trust old network-node IDs

`center_network_node` is an implementation-local integer from a particular augmented road graph. It is not a portable geographic identifier. The restart therefore uses the saved **median geometries**, snaps those medians to the supplied road linework, inserts them into a fresh road graph, and computes median-to-median road distances there. This keeps the saved medians and partitions reusable without depending on a hidden checkpoint graph or its old node numbering.

## Invariants enforced as errors

A normal run must satisfy:

```text
number of retained clusters
= number of network centers
= number of partitions
= number of X nodes
```

The engine also refuses to continue when:

- the transaction and technical-coefficient matrices have different sector sets;
- a custom transaction workbook is asked to derive A but has no unambiguous positive `Total Output` column aligned with the IO rows;
- a retained spatial type is missing from the IO matrix;
- MWAS does not return a DAG;
- a retained positive transaction edge has a non-positive technical coefficient;
- center/partition/point cluster keys disagree;
- a partition is empty, non-polygonal, invalid, or has non-positive area;
- a point-to-own-median distance is missing, negative, non-finite, or a legacy Euclidean fallback;
- a restart point identifier is blank or duplicated;
- the final point score violates its configured bounded attenuation interval.

There is no Euclidean center fallback and no Euclidean Voronoi fallback in the revised full workflow. A failure in an exact required object is surfaced instead of silently changing the method.

## Centrality

Network X remains directed in `sigma_X_edges.csv` for audit, but direction is removed before centrality. SIGMA computes weighted eigenvector centrality on the positive-weight undirected projection of X. The legacy `--centrality-direction` option is still accepted for compatibility but does not alter the result. Run metadata records `centrality_method=undirected_weighted_eigenvector` and `centrality_direction=undirected`.

## Source layout

```text
src/sigma_engine/
  io_workflow.py    # Z -> MWAS edge set -> A reweighting
  scoring.py        # cluster-p90 bounded point score
  pipeline.py       # seven-step orchestration + restart from Step 4
  cli.py            # run / from-partitions
```

Existing spatial primitives (`network.py`, `clustering.py`, `center.py`, `voronoi.py`, `spatial_graph.py`, sparse solvers, packaged IO resources) remain in place.

See `docs/IMPLEMENTATION_NOTES.md` for the current-repository audit and exact migration map, and `docs/SIGMA_REVISED_WORKFLOW.md` for the specification used for this reset.
