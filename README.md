# sigma-engine

`sigma-engine` is the focused implementation of the SIGMA spatial-economic centrality
workflow. It consumes classified establishment points, a projected road network, and a polygon
boundary. The economic network comes from the bundled 2018 PSA IO80 transaction matrix by
default (or bundled IO16 when selected), so normal runs require no external IO file. The engine
then writes a GeoParquet in which every retained point has a road-network/spatial economic
centrality score.

Point production is deliberately outside `sigma-engine`. The engine starts from an existing
classified point vector and implements only network preparation, HDBSCAN, cluster centers,
Voronoi surfaces, MWAS IO-DAG construction, spatial instantiation, centrality, scoring, and output
validation. This keeps point producers and the analysis engine independently runnable.

## Workflow and invariants

```text
classified point vector (`--points`)
        |
        v
auto-detect/override IO80 [default] or IO16 point field -> normalize as `type`
        |
        +---- bundled PSA 2018 IO80 [default] / IO16 transaction matrix
        v
continuous point-to-road snapping -> one shared augmented road graph
        |
        v
sparse network-distance HDBSCAN independently by type
        |  on recoverable failure: Euclidean HDBSCAN within each road component
        |  if fallback also fails: skip that type
        +---- noise (-1) is removed
        v
one node for every unique [type, cluster]
        |
        +---- exact network-node 1-median
        |      on recoverable failure: Euclidean centroid -> nearest same-component road -> nearest edge node
        |      if fallback also fails: skip that cluster
        +---- per-type network Voronoi -> explicit 2-D surface approximation
               on coarse-grid failure: refine that type only (6 network attempts total)
               if all network attempts fail: clipped Euclidean Voronoi
               if that also fails: skip that type
        v
weighted IO graph -> fast MWAS heuristic (default) -> weighted IO DAG
                  or exact HiGHS MWAS (--mwas-method exact)
        |
        v
instantiate each retained IO-DAG edge where type partitions overlap
        |
        +---- dag_weight
        +---- road_distance(center_1, center_2)
        +---- weight = dag_weight * road_distance
        v
weighted directed eigenvector centrality on X
        |
        v
point score = EC / road_distance(point, cluster_center)
        |
        v
validated GeoParquet + audit CSV/JSON outputs
```

The implementation checks the following cross-stage invariants explicitly:

- one existing classified point vector is supplied with `--points`;
- point `canonical_id` values are non-blank and unique after classification filtering;
- all selected point `type` values exist in the selected built-in/override IO matrix;
- the IO transaction block has identical normalized row/column sector sets;
- sparse HDBSCAN neighbour graphs contain only validated finite, symmetric road distances;
- every retained `[type, cluster]` has exactly one network center and one rendered partition;
- the spatial network `X` remains acyclic because it inherits directions from the MWAS IO DAG;
- all point-to-center distances, eigenvector values, and final scores are finite;
- every GeoParquet is reopened with GeoPandas immediately after writing.

## Installation

Python 3.11+ is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

No point producer is a runtime dependency of `sigma-engine`. Produce/classify POIs first, then
pass the resulting vector file to the engine with `--points`.

## Inputs

### 1. Classified points

`sigma-engine` consumes an existing GeoPandas-readable point vector. It does not launch,
import, or configure a point producer.

The narrow required contract is:

```text
canonical_id
geometry
CRS
one unambiguous IO80 or IO16 classification field
```

`canonical_name` is retained when present. If a compact GIS product omits it, the engine uses
`name` when available and otherwise falls back to the textual `canonical_id`; names do not
participate in any algorithm.

For `--classification io80`, automatic classification-field detection recognizes:

- `io80_code` — compatible with `sigma-siphon` `pois.parquet`;
- `io80_map_code` — compatible with the map-safe `placetype-ph` classified product.

For `--classification io16`, the recognized fields are `io16_code` and `io16_map_code`.
If both recognized fields coexist, they must agree wherever both are populated; the more
complete field is then used. A conflict fails loudly rather than silently choosing one.

Plural candidate-set fields such as `io80_codes` and `io16_codes` are deliberately **not**
auto-selected because they may encode ambiguous mappings rather than one sector per point.
For another producer or custom schema, explicitly name the single-valued field with
`--io80-column` or `--io16-column`.

Spreadsheet-style numeric sector IDs are normalized consistently. For example, `1`, `1.0`,
`"1"`, and `"01"` all become the internal type `"01"`. Blank classifications are excluded
before clustering and counted in run metadata.

### 2. Roads

Roads may be any GeoPandas-readable line dataset, including GeoPackage/Shapefile or
GeoParquet. Their CRS must be projected; every road distance is expressed in that CRS's
linear units.

Consecutive coordinates become undirected graph arcs. Two lines connect only when their
canonicalized source vertices are identical. A 2-D geometric crossing does **not** invent a
junction. Point positions are snapped continuously onto road arcs and then inserted into one
shared augmented graph used by all subsequent algorithms.

`snap_distance_to_network` is straight-line QA metadata. It is not added to road-network
distance. Use optional `--max-snap-distance` to fail a run when a classified point is
implausibly far from the supplied roads.

### 3. Boundary

The boundary must be polygonal and have a declared CRS. It is reprojected to the road CRS and
used only to render the 2-D per-type Voronoi surfaces.

### 4. Built-in IO transaction matrix

Normal runs do **not** require an IO-table argument. `sigma-engine` owns the economic-network
specification and ships the normalized 2018 Philippine Statistics Authority benchmark
transaction matrices as package resources:

- `--classification io80` (default) -> bundled 2018 PSA 80x80 transaction matrix;
- `--classification io16` -> bundled 2018 PSA 16x16 transaction matrix.

The packaged files contain only the square intermediate transaction blocks needed by the
engine. Rows are **suppliers**, columns are **users**, values are in million Philippine pesos,
and sector positions are normalized to canonical `01..80` or `01..16`. A positive cell
`z[i,j]` therefore means:

```text
type_i -> type_j
```

The package verifies each bundled resource by SHA-256 before loading it. Run metadata records
the source ID, reference year, PSA source URL, and resource hash so the economic network is
reproducible. The bundled IO16 and IO80 matrices have the same aggregate intermediate-use
total, as required by aggregation.

Source: Philippine Statistics Authority, *PSA releases the 2018 Input-Output Tables*,
Reference No. 2021-510, released 9 December 2021:
https://psa.gov.ph/content/psa-releases-2018-input-output-tables

The PSA page states that the 2018 benchmark accounts include transaction tables at 16x16,
80x80, and 240x240 resolution and that PSA website data/content are CC BY 4.0 unless otherwise
stated. See `src/sigma_engine/resources/NOTICE.txt` for the bundled-data notice.

#### Custom IO override

For a deliberate experiment, `--io-table-override PATH` replaces the built-in matrix for that
run. Supported override formats are CSV, `.xlsx`/`.xlsm`, and Parquet. `--io-sheet` applies
only to an Excel override. Legacy `.xls` remains unsupported.

For Excel overrides, SIGMA accepts either a clean square matrix or the common publication
layout in which titles/descriptions/totals/final-demand columns surround the intermediate
transaction block. In IO16/IO80 mode the engine locates the matching 16x16 or 80x80 block and
maps the published sector order to canonical SIGMA codes.

Positive diagonal entries remain part of the source IO network for faithful auditing. A
self-loop can never belong to a DAG, so every MWAS method necessarily removes it from the
retained acyclic graph. Values must be finite and non-negative, and normalized row/column
sector sets must match exactly. No economic edge threshold is applied before MWAS.

## Sparse network-distance HDBSCAN*

Clustering runs independently for every `type` using shortest-path distance on the supplied
road network. SIGMA does **not** build an all-pairs distance matrix. Points are snapped
vectorially, coincident snapped positions are compressed with multiplicity, and bounded local
shortest-path searches materialize only neighbour pairs inside the current search radius.
HDBSCAN* is then evaluated directly on that sparse distance graph. Road vertices are
canonicalized to 11 significant digits by default before graph construction; geometric crossings
without a shared source vertex are not automatically connected.

SIGMA uses `adaptive` distance search by default as an engine-level convenience. The bounded
search starts below `--hdbscan-max-distance`, grows geometrically by 1.5, and may stop when two
consecutive successful trials have the same flat clustering/membership and no more than 1% of
otherwise-reachable core distances remain truncated. The default ceiling is 5,000 road-CRS
units. This ceiling is a SIGMA default, not an inferred property of the data. Use
`--hdbscan-distance-mode fixed` when a study requires one explicit truncation horizon.
`--hdbscan-max-neighbor-pairs` is a memory guard on stored sparse pairs, not on the number of
observations.

Every run writes `sigma_clustering_summary.csv` and
`sigma_clustering_distance_trace.csv`. The first records the final selected radius, status, pair
count and truncation diagnostics for each economic type. The second records every distance trial,
so an adaptive decision remains auditable. A status of `converged` means local flat-result
stability on the tested radius ladder; it is not a proof that every larger radius would produce
the same clustering.

This design makes memory scale with measured neighbour pairs rather than `n^2`. If the sparse
network-HDBSCAN calculation raises a recoverable per-type error, fail-soft mode retries that
type with scikit-learn's ordinary Euclidean HDBSCAN on projected XY coordinates. The fallback
is still run separately inside each connected road component, so it cannot create a cluster
that crosses an unreachable road-network break. If the Euclidean fallback also fails, that type
is skipped. Every fallback or skip is recorded in `sigma_stage_events.csv`. `--fail-fast`
disables these recoveries and restores immediate-abort behavior.

Noise (`cluster = -1`) is dropped. Remaining labels are renumbered deterministically within each
`type`; therefore `cluster` is only unique together with `type` (or `node_id`).

## Network 1-median

Every retained `[type, cluster]` uses all member points as unit-weight demand. Repeated points
at one network node are compressed into exact multiplicities. The solver uses a SciPy
demand-by-node shortest-path reduction in bounded memory blocks rather than Python per-node
accumulation.

The chosen center minimizes:

```text
sum_p d_road(candidate, p)
```

over vertices of the relevant augmented road component. Ties go to the smallest stable network-node ID. If the exact network 1-median fails for one
cluster in fail-soft mode, SIGMA computes that cluster's Euclidean centroid, finds the nearest
road on the same connected road component, and uses the nearer existing endpoint of that
already-split road edge as the fallback network node. This keeps downstream shortest-path and
Voronoi stages well-defined. The centroid, continuous road-snap coordinates, snap distances,
and fallback Euclidean objective are preserved in the center output. If that fallback also
fails, only that cluster is skipped. One result is written per surviving cluster with:

```text
type
cluster
node_id
center_network_node
median_objective
center_method
fallback_* audit fields
geometry
```

## Network Voronoi and 2-D surface

For one type at a time, every retained point is a source labelled by its cluster. A SciPy
multi-source shortest-path search assigns road nodes to the nearest cluster; deterministic tie
correction gives exact equal-distance ties to the smaller cluster ID.

The exact partition is one-dimensional on the road network. Spatial overlap between different
types requires polygons, so SIGMA renders an explicit boundary grid:

1. construct and clip the regular grid with vectorized Shapely operations;
2. take vectorized point-on-surface anchors;
3. attach all anchors in one spatial-index query to the nearest **source-reachable road component for that type**;
4. inherit the exact network-Voronoi owner at that road location;
5. dissolve cells by `[type, cluster]`.

Source-less road components are excluded from a type's surface attachment because network
distance to that type is undefined there. This avoids silently leaving holes in what is meant
to be a partitioned map.

`--voronoi-resolution` controls the initial approximation. If it is so coarse that a retained
cluster receives no grid cell, SIGMA automatically retries that **type only** at progressively
finer resolutions. The default is five refinements after the initial attempt: **six network-
Voronoi attempts total** at resolutions `r, 0.5r, 0.25r, ...`.

If all network attempts fail, SIGMA then falls back to an ordinary **Euclidean Voronoi**
partition using one final cluster-center coordinate per `[type, cluster]`, clipped to the study
boundary. If two clusters have exactly the same final center, only those duplicate sites are
replaced by their raw projected cluster centroids; this substitution is recorded rather than
silently jittering coordinates. The partition output records `surface_method` and
`surface_seed_method`, while the planar fallback has `surface_resolution = NaN` because it is
not grid-based.

Only if the Euclidean fallback also fails does the default fail-soft policy record and skip that
type. `sigma_stage_events.csv` and the end-of-run adjustment summary report every refinement,
Euclidean fallback, duplicate-center seed substitution, and final skip. Use `--fail-fast` to
abort only after the recovery chain itself fails. `--voronoi-max-cells` remains a memory/work
guard for the network-grid attempts; hitting it moves that type to the Euclidean fallback rather
than killing unrelated work.

## IO DAG and MWAS

The weighted directed IO graph is reduced to an acyclic subgraph before spatial
instantiation. The objective underlying exact mode is the **maximum-weight acyclic
subgraph (MWAS)** problem, equivalently the complement of a minimum-weight feedback
arc set. The weighted problem is NP-hard in general (Hassin & Rubinstein, 1994).

The production default is deliberately the fast deterministic heuristic:

```text
sigma-engine run ... --mwas-method fast   # default
```

It sorts positive non-self IO edges by descending transaction weight and accepts an
edge exactly when adding it would not create a directed cycle. Equal-weight ties are
resolved deterministically by sector IDs. This produces a maximal acyclic subgraph
quickly, but it is **not guaranteed globally maximum-weight**; SIGMA records
`mwas_optimal=false` for this mode. The specific descending-weight greedy rule is a
package engineering heuristic, and no approximation ratio is claimed for it.

Exact mode is explicit:

```text
sigma-engine run ... --mwas-method exact
```

Every IO type receives an integer topological rank and every eligible edge receives a
binary keep variable. Selecting `u -> v` enforces:

```text
rank(u) + 1 <= rank(v)
```

while the objective maximizes total retained IO weight. SciPy's MILP interface uses
HiGHS with zero relative MIP gap. Exact mode never silently downgrades: if a proven
optimum is not returned, the run fails explicitly.

Algorithmic background: Hassin & Rubinstein (1994), *Approximations for the maximum
acyclic subgraph problem*, Information Processing Letters 51(3), 133--140; and Charon
& Hudry (2006), *A branch-and-bound algorithm to solve the linear ordering problem for
weighted tournaments*, Discrete Applied Mathematics 154(15), 2097--2116. The latter
discusses the equivalence among maximum acyclic subdigraph, minimum feedback arc set,
and linear-ordering formulations.

SIGMA writes `sigma_io_dag.csv` immediately after MWAS and records source/retained/removed
edge counts and weights, the requested method, and whether optimality was proven in the
run metadata.

## Spatial network X

Every unique `[type, cluster]` is created as an `X` node before edges are considered, so
isolated clusters are retained.

For each IO-DAG edge `type_1 -> type_2`, each `type_1` partition is compared with intersecting
`type_2` partitions. Boundary-only contact is insufficient; the polygon intersection must
have positive area.

Each qualifying pair creates:

```text
[cluster_1, type_1] -> [cluster_2, type_2]
```

with three separate edge attributes:

```text
dag_weight        # retained IO edge weight from the MWAS DAG
road_distance
weight = dag_weight * road_distance
```

The multiplicative distance behavior follows the requested SIGMA definition literally: a
larger road distance increases `weight` rather than penalizing it.

If two centers are disconnected on the road graph, the edge is not created and the pair is
written to `sigma_disconnected_overlap_pairs.csv`.

Center-to-center road distances are evaluated in a bounded-memory source-center sweep. For
one source road node, SIGMA computes one SciPy shortest-path vector, extracts only the target
centers required by spatial overlaps across all of that sector's outgoing IO-DAG edges, and
discards the full vector immediately. Source clusters sharing the same center road node reuse
that one computation. SIGMA therefore does not retain an all-road-node distance dictionary for
every center while constructing `X`.

## Directed eigenvector centrality and the DAG consequence

The direction convention is explicit:

- `incoming` (default): prestige arrives from important predecessors;
- `outgoing`: apply the same definition to the reversed graph.

Because every spatial edge inherits a direction from an IO **DAG**, `X` is also a DAG. A DAG
adjacency matrix is nilpotent, so every eigenvalue is zero and ordinary positive Perron
eigenvector centrality is non-unique.

SIGMA records this rather than disguising it. For a DAG it returns one deterministic,
non-negative unit-norm zero-eigenvalue vector:

- equal mass on positive-weight terminal nodes for `incoming`;
- equal mass on positive-weight initial nodes for `outgoing`;
- zero elsewhere.

A zero-weight edge (possible when two network centers coincide) is retained in the edge audit
but does not alter the centrality support because it contributes a zero adjacency entry.

Run metadata sets `eigenvector_degenerate_dag=true` and explains the convention. If a future
version wants non-degenerate recursive prestige on a DAG, that should be introduced as a
separately named measure rather than silently changing the meaning of eigenvector centrality.

## Point centrality score and zero road distance

For a normal network-distance row:

```text
centrality_score = eigenvector_centrality / network_distance_to_center
```

If a cluster's shortest-path point-to-center calculation fails under fail-soft mode, the same
column carries projected Euclidean point-to-center distance for that recovered cluster and
`point_center_distance_method` is set to `euclidean_fallback`. Thus the numeric score always
remains defined, while the metric actually used is explicit in the principal output and audit
metadata.

A retained point can coincide exactly with its cluster center. The literal formula would then
be singular. SIGMA therefore uses the explicit finite rule:

```text
centrality_score = eigenvector_centrality / zero_distance_floor
```

**only** when the road distance is exactly zero. `zero_distance_floor` defaults to `1.0` in
the road CRS's linear units.

## Command line

Point production/classification happens first in whichever upstream application you choose.
Then run the engine on that product:

```bash
sigma-engine run \
  --points classified_points.parquet \
  --roads roads.gpkg \
  --boundary boundary.gpkg \
  --output-dir output \
  --voronoi-resolution 500
```

No `--area` or producer-specific options exist in `sigma-engine`. IO80 is the default economic
resolution and uses the bundled PSA 2018 IO80 matrix automatically. For the established producer
schemas, point classification columns are detected automatically. Use `--classification io16`
for bundled IO16, `--io80-column FIELD` / `--io16-column FIELD` for a custom point field, and
`--io-table-override PATH` only when intentionally replacing the built-in economic matrix.

## Durable checkpoints and resume

Long runs write durable stage artifacts immediately instead of waiting for the final score.
Resume is enabled by default. If the process or Python session crashes, rerunning the same
`sigma-engine run ...` command reuses matching completed stages rather than starting from
clustering again.

A checkpoint is reused only when the engine version, relevant run configuration, and the
resolved input file path/size/mtime all match. A changed points, roads, boundary, IO override,
or modeling setting invalidates the old checkpoint automatically. Internal restart state lives
in `.sigma_checkpoints/manifest.json` inside the output directory.

The durable stages are:

1. clustering;
2. network 1-median centers;
3. point-to-center distances;
4. Voronoi partitions; and
5. spatial network `X`.

The sparse road graph is rebuilt on resume because it is cheap relative to these stages and
keeps the checkpoint representation portable. The expensive completed stage itself is not
recomputed.

Use `--no-resume` to recompute all stages from scratch while continuing to write new
checkpoints, or `--restart` to explicitly remove prior SIGMA artifacts/checkpoints in the
output directory before recomputing.

Stage artifacts are written as soon as their stage succeeds:

- `sigma_clustered_points.parquet` after clustering;
- `sigma_clustering_summary.csv` and `sigma_clustering_distance_trace.csv` after clustering;
- `sigma_network_centers.parquet` after 1-median solving;
- `sigma_points_with_center_distance.parquet` after point-center distances;
- `sigma_partitions.parquet` and `sigma_points_partitioned.parquet` after Voronoi;
- `sigma_io_dag.csv` immediately after MWAS; and
- `sigma_X_nodes.csv` / `sigma_X_edges.csv` immediately after network X construction; and
- `sigma_disconnected_overlap_pairs.csv` immediately after network X construction (header-only
  when there are no disconnected overlap pairs).

`sigma_stage_events.csv` is refreshed after every durable stage, so recoveries/skips remain
auditable even if a later stage crashes. Final outputs overwrite/complete the corresponding
major-stage artifacts where appropriate.

## Outputs

### Principal GeoParquet

`sigma_points_centrality.parquet` contains:

```text
canonical_id
canonical_name
type
cluster
node_id
network_component
snapped_edge_id
snap_distance_to_network
network_distance_to_center
point_center_distance_method
eigenvector_centrality
centrality_score
geometry
```

Geometry is written in the projected road CRS used by the analysis.

### Geospatial audit artifacts

- `sigma_clustered_points.parquet`
- `sigma_network_centers.parquet`
- `sigma_points_with_center_distance.parquet`
- `sigma_partitions.parquet`
- `sigma_points_partitioned.parquet`

Both include `type`, `cluster`, and the same `node_id` used by network `X`, so joins do not
require reconstructing composite keys.

### Tabular audit artifacts

- `sigma_clustering_summary.csv`
- `sigma_clustering_distance_trace.csv`
- `sigma_stage_events.csv` (recoveries and skipped independent units)
- `sigma_io_dag.csv`
- `sigma_X_nodes.csv`
- `sigma_X_edges.csv`
- `sigma_disconnected_overlap_pairs.csv` when applicable
- `sigma_run_metadata.json`

CSV files keep stable column headers even when they contain zero rows. The clustering summary
records one final adaptive/fixed-distance diagnostic row per economic type; the distance trace
records every attempted radius. `sigma_stage_events.csv` records every automatic fallback, Voronoi refinement, and skipped
independent unit. Network HDBSCAN falls back to component-separated Euclidean HDBSCAN; failed
network 1-medians fall back to a Euclidean centroid snapped back to the same-component road
network; failed point-to-center shortest-path distances fall back per cluster to projected
Euclidean distance. Network Voronoi gets six attempts by default, then falls back to a clipped
planar Euclidean Voronoi; only if that fallback also fails is the type skipped. If any fallback
also fails, only that independent type/cluster is skipped. All adjustment records are repeated
at the end of a progress-enabled run and are also embedded in `sigma_run_metadata.json`.
Metadata distinguishes HDBSCAN noise from points dropped after recoverable stage failures,
records the Voronoi method used by each surviving type, and records the actual network-grid
resolution where applicable. It also records primary/fallback method counts, road graph sizes,
snap-distance QA,
clustering status counts and selected radii, IO/MWAS counts and retained weight, `X` counts,
centrality semantics, and the complete serialized run configuration.

GeoParquet outputs are immediately reopened by GeoPandas after writing. Missing CRS metadata
or a changed row count therefore fails the run at the producer rather than surfacing later in
GeoPandas or QGIS.

## Development checks

```bash
python -m compileall -q src tests
python -m pytest -q
ruff check .
```

The test suite includes regression coverage for IO-axis mismatches, non-finite IO weights,
continuous snapping, duplicate road arcs, fast and exact MWAS DAG invariants, zero-weight centrality edges,
source-less road components, automatic coarse-grid refinement, Euclidean Voronoi fallback, fail-soft per-type/per-cluster
continuation, Euclidean HDBSCAN and center/distance fallbacks, road-component separation of
fallback clusters, producer-schema normalization, checkpoint/resume behavior, and the
GeoParquet end-to-end path when `pyarrow` is available.
