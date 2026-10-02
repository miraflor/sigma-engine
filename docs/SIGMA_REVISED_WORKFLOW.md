# SIGMA Engine — Revised Workflow Specification

**Status:** Revised workflow handoff for a clean restart
**Target reset version:** `v0.1.0`
**Primary classification:** IO80
**Default IO acyclic transformation:** Fast MWAS
**Purpose:** Define the intended SIGMA workflow precisely before implementation.

## 1. Core data hierarchy

SIGMA is organized around:

\[
\text{type } s
\rightarrow
\text{clusters } C_{s,1},\ldots,C_{s,m_s}
\rightarrow
\text{points within each cluster}.
\]

For type \(s\), cluster \(c\) contains

\[
C_{s,c}=\{p_{s,c,1},\ldots,p_{s,c,n_{s,c}}\}.
\]

Each retained cluster has exactly:
1. one HDBSCAN cluster identity \((s,c)\);
2. one network 1-median \(M_{s,c}\);
3. one polygonal partition shape \(P_{s,c}\);
4. one spatial-economic graph node \(X_{s,c}\).

The 1-median and the partition are two different representations of the same cluster. They are produced independently from the clustered points and serve different downstream purposes.

## 2. High-level dependency graph

```text
Classified points
      |
      v
Network HDBSCAN by type
      |
      +-------------------------------+
      |                               |
      v                               v
Network 1-median branch          Cluster-partition branch
(one median per cluster)         (one shape per cluster)
      |                               |
      |                               |
      +------------+------------------+
                   |
                   v
          Spatial-economic graph X
                   ^
                   |
Transactions table -> fast MWAS -> surviving DAG edges
                                  -> reweight with technical coefficients
                   |
                   v
            Cluster centrality
                   |
                   v
             Point-level score
                   ^
                   |
        point-to-own-median distance
```

Key dependencies:
- Step 3 and Step 4 are parallel outputs of Step 2.
- Step 3 is used in Steps 6 and 7.
- Step 4 is used in Step 6.
- Step 5 supplies the allowed directed type relationships and their final technical-coefficient weights to Step 6.

# 3. Step-by-step workflow

## Step 1 — Classified points and road-network positioning

### Input
A point GeoParquet containing already-classified establishments/POIs.

At minimum:
- unique point identifier;
- point geometry;
- IO classification, normally IO80;
- classification field such as `io80_map_code`.

Additional spatial inputs:
- road network;
- study boundary.

### Processing
For every type \(s\):
1. collect all points whose IO classification is \(s\);
2. validate geometries and CRS;
3. snap each point to the road network for later network-distance operations;
4. retain the original observed point geometry.

If type \(s\) initially has \(n_s\) points, denote them

\[
p_{s,1},p_{s,2},\ldots,p_{s,n_s}.
\]

### Output
Network-located classified points.

Conceptual fields:

```text
point_id
type
geometry
network_position
snap_distance
```

This output feeds Step 2.

## Step 2 — Network HDBSCAN within each type

### Input
For one type \(s\):
- all points of type \(s\);
- their road-network positions;
- road network;
- HDBSCAN parameters.

### Processing
Run network HDBSCAN separately for each type.

For type \(s\), the retained observations are divided into

\[
C_{s,1},C_{s,2},\ldots,C_{s,m_s},
\]

plus possible noise.

Each retained cluster contains its own member points:

\[
C_{s,c}=\{p_{s,c,1},\ldots,p_{s,c,n_{s,c}}\}.
\]

Noise is assigned `cluster = -1` and does not proceed to cluster-level SIGMA calculations.

### Output
Clustered points keyed by:

```text
[type, cluster]
```

Conceptual fields:

```text
point_id
type
cluster
geometry
network_position
```

The Step-2 output branches independently into Steps 3 and 4.

## Step 3 — Network 1-median for each cluster

### Input
For one retained cluster

\[
C_{s,c}=\{p_{s,c,1},\ldots,p_{s,c,n_{s,c}}\},
\]

use:
- the points inside that cluster;
- their road-network positions;
- the road network.

### Processing
Compute the exact network 1-median

\[
M_{s,c}
=
\arg\min_v \sum_i d_G(v,p_{s,c,i}),
\]

where \(d_G\) is shortest-path distance on the road network.

Then compute each cluster member's road-network distance to that median:

\[
d_{s,c,i}=d_G(p_{s,c,i},M_{s,c}).
\]

### Output
For every cluster \((s,c)\):

**Cluster-center output**
```text
type
cluster
center_geometry
center_network_node
```

representing \(M_{s,c}\).

**Point-to-center output**
```text
point_id
type
cluster
distance_to_cluster_median
```

representing \(d_G(p_{s,c,i},M_{s,c})\).

### Downstream use
- **Step 6:** \(M_{s,c}\) is used to calculate road-network distances between connected cluster nodes.
- **Step 7:** point-to-own-median distances mildly differentiate points within the same cluster.

## Step 4 — One spatial partition shape per cluster

### Input
For one type \(s\):
- all retained Step-2 points of type \(s\);
- each point's cluster label;
- road network;
- study boundary.

Conceptually:

```text
type s
  cluster 1
    point 1
    point 2
    ...
  cluster 2
    point 1
    point 2
    ...
  ...
  cluster m_s
```

### Processing
Process each type independently.

For type \(s\):
1. use every retained point of type \(s\) as a network-Voronoi site;
2. assign road-network territory to the nearest retained point according to the network-Voronoi procedure;
3. each point-level Voronoi territory carries the HDBSCAN cluster label of its generating point;
4. merge the **polygonal Voronoi territories** carrying the same cluster label;
5. clip the resulting geometry to the study boundary;
6. produce one final Polygon or MultiPolygon for each retained cluster.

For cluster \(C_{s,c}\), the final spatial representation is

\[
P_{s,c},
\]

a Polygon or MultiPolygon.

### Output
Exactly one partition geometry per retained cluster:

```text
type
cluster
geometry
```

For type \(s\), if HDBSCAN produced \(m_s\) retained clusters, Step 4 produces:

\[
P_{s,1},P_{s,2},\ldots,P_{s,m_s}.
\]

Therefore:

\[
\text{number of retained clusters}
=
\text{number of cluster partitions}.
\]

Example:

```text
type   cluster   geometry
55     1         Polygon/MultiPolygon
55     2         Polygon/MultiPolygon
...
55     808       Polygon/MultiPolygon
56     1         Polygon/MultiPolygon
...
```

Different types are partitioned independently, so shapes belonging to different types may overlap.

### Intended artifact
```text
sigma_partitions.parquet
```

One row per retained `[type, cluster]`.

## Step 5 — IO DAG: MWAS selection from transactions, final weights from technical coefficients

Step 5 intentionally uses two IO matrices for two different purposes.

### Input A — Transactions table
Let

\[
Z=[z_{ab}]
\]

be the IO transactions matrix.

### Processing A — MWAS edge selection
1. construct the directed weighted sector graph from \(Z\);
2. use transaction values \(z_{ab}\) as the MWAS weights;
3. apply **fast MWAS by default**;
4. MWAS determines which directed sector edges survive while making the graph acyclic.

This produces the surviving edge set:

\[
E_{\text{MWAS}}.
\]

Transactions determine **which edges survive**.

### Input B — Technical coefficients from total output or explicit A
Let

\[
A=[A_{ab}]
\]

be the technical-coefficient matrix. Under the usual IO orientation:

\[
A_{ab}=\frac{z_{ab}}{x_b},
\]

where \(x_b\) is total output of sector \(b\).

For the built-in PSA IO80 and IO16 configurations, the engine ships the aligned **Total Output** vector with the transaction block and derives \(A\) directly from \(Z\) and \(x\). A full custom transaction workbook may instead supply its own `Total Output` column, and a separately supplied technical-coefficient matrix remains a valid override. The intermediate-demand column sum is **not** a substitute for total output.

### Processing B — Reweight surviving DAG edges
For every surviving MWAS edge

\[
a\rightarrow b\in E_{\text{MWAS}},
\]

assign the corresponding technical coefficient:

\[
w_{ab}=A_{ab}.
\]

### Output
A directed acyclic IO graph:

\[
G_{\text{IO-DAG}}=(V,E_{\text{MWAS}})
\]

whose downstream edge weights are:

\[
\boxed{w_{ab}=A_{ab}}.
\]

In words:

\[
\boxed{\text{Transactions determine which edges survive MWAS}}
\]

and

\[
\boxed{\text{Technical coefficients determine the final DAG edge weights}}.
\]

### Default and optional MWAS modes
Default:

```text
fast
```

Optional explicit mode:

```text
exact
```

### Intended artifact
```text
sigma_io_dag.csv
```

Conceptual fields:

```text
source_type
target_type
transaction_value
technical_coefficient
mwas_method
```

The graph edge weight used downstream is `technical_coefficient`.

## Step 6 — Spatial-economic graph X

### Inputs
Step 6 combines:
1. cluster partition shapes \(P_{s,c}\) from Step 4;
2. cluster 1-medians \(M_{s,c}\) from Step 3;
3. final IO DAG from Step 5;
4. road network.

### Node definition
Every retained cluster becomes exactly one X node:

\[
X_{s,c}.
\]

Therefore:

\[
\text{number of X nodes}
=
\text{number of retained clusters}.
\]

### Processing
Take one retained IO-DAG edge:

\[
a\rightarrow b
\]

with final technical-coefficient weight:

\[
A_{ab}.
\]

#### 6.1 Compare source and target cluster partitions
Consider every source cluster partition:

\[
P_{a,1},\ldots,P_{a,m_a}
\]

and every target cluster partition:

\[
P_{b,1},\ldots,P_{b,m_b}.
\]

For each pair \((i,j)\), test:

\[
\operatorname{Area}(P_{a,i}\cap P_{b,j})>0.
\]

Only positive-area overlap qualifies. A zero-area boundary touch does not qualify.

#### 6.2 Instantiate the directed cluster pair
If the overlap is positive:

\[
X_{a,i}\rightarrow X_{b,j}
\]

is eligible.

Its direction comes from the IO DAG.

#### 6.3 Compute road distance from the Step-3 medians
Retrieve:

\[
M_{a,i},\quad M_{b,j}.
\]

Compute:

\[
d_{ij}=d_G(M_{a,i},M_{b,j}).
\]

If the medians are disconnected on the road network, record the overlap separately and do not create a usable X edge.

#### 6.4 Assign the spatial-economic edge weight
For a road-connected qualifying pair:

\[
\boxed{
w_{(a,i),(b,j)}
=
A_{ab}\,d_{ij}
}.
\]

Thus the X edge combines:
- economic intensity from \(A_{ab}\);
- spatial information from median-to-median road distance \(d_{ij}\).

### Example
Suppose MWAS retains:

\[
55\rightarrow62
\]

and after reweighting:

\[
A_{55,62}=0.18.
\]

Suppose:

\[
\operatorname{Area}(P_{55,3}\cap P_{62,7})>0.
\]

Then the eligible cluster relationship is:

\[
X_{55,3}\rightarrow X_{62,7}.
\]

If:

\[
d_G(M_{55,3},M_{62,7})=2400\text{ m},
\]

then:

\[
w=0.18\times2400=432.
\]

### Centrality
After all eligible cluster edges are instantiated, compute the selected directed weighted centrality on \(X\).

### Output

**X nodes**
```text
sigma_X_nodes.csv
```

Conceptual fields:

```text
type
cluster
centrality
```

**X edges**
```text
sigma_X_edges.csv
```

Conceptual fields:

```text
source_type
source_cluster
target_type
target_cluster
technical_coefficient
road_distance
edge_weight
```

where:

\[
\text{edge_weight}
=
\text{technical_coefficient}
\times
\text{road_distance}.
\]

**Disconnected overlaps**
```text
sigma_disconnected_overlap_pairs.csv
```

for positive-area partition overlaps whose cluster medians are disconnected on the road network.

## Step 7 — Point-level SIGMA score with tempered distance effect

### Objective
All points in the same cluster inherit the same cluster-level economic-spatial centrality.

Distance from the cluster median should differentiate points within that cluster, but only mildly. Cluster membership should remain the dominant source of score differences.

The old rule

\[
\frac{C_{s,c}}{d}
\]

is too aggressive because it can create very large within-cluster variation.

### Inputs
For point \(p_{s,c,i}\), use:

1. cluster-node centrality
   \[
   C_{s,c};
   \]

2. point-to-own-cluster-median road distance
   \[
   d_{s,c,i}=d_G(p_{s,c,i},M_{s,c}).
   \]

### Recommended normalization
For each cluster, calculate:

\[
R_{s,c}
=
Q_{0.90}
(d_{s,c,1},\ldots,d_{s,c,n_{s,c}}),
\]

the 90th percentile of member-point distances to the cluster median.

Define bounded normalized distance:

\[
q_{s,c,i}
=
\min
\left(
\frac{d_{s,c,i}}{\max(R_{s,c},1)},
1
\right).
\]

Thus:

\[
0\le q_{s,c,i}\le1.
\]

### Recommended tempered point-score rule

\[
\boxed{
S_{s,c,i}
=
C_{s,c}
\left(
1-\lambda q_{s,c,i}
\right)
}
\]

with default:

\[
\boxed{\lambda=0.15}.
\]

Expose \(\lambda\) as a configurable parameter:

```text
distance_tempering = 0.15
```

### Interpretation
At the cluster median:

\[
q=0
\]

and:

\[
S=C_{s,c}.
\]

At or beyond the cluster's 90th-percentile distance:

\[
q=1
\]

and, with \(\lambda=0.15\):

\[
S=0.85C_{s,c}.
\]

Therefore:

\[
\boxed{
0.85C_{s,c}
\le
S_{s,c,i}
\le
C_{s,c}
}
\]

under the default.

This caps within-cluster distance variation at 15%, keeping cluster centrality dominant.

Suggested sensitivity range:

```text
0.00 = no within-cluster distance differentiation
0.10 = very mild
0.15 = recommended default
0.20 = moderate
0.25 = stronger but still bounded
```

### Output
```text
sigma_points_centrality.parquet
```

Conceptual fields:

```text
point_id
type
cluster
cluster_centrality
distance_to_cluster_median
cluster_distance_p90
normalized_cluster_distance
distance_tempering
sigma_score
geometry
```

with:

\[
\text{sigma_score}
=
C_{s,c}(1-\lambda q_{s,c,i}).
\]

# 4. Exact object relationships

For each retained cluster:

```text
C_{s,c}
  |
  +-- contains --> points p_{s,c,i}
  +-- has exactly one --> network 1-median M_{s,c}
  +-- has exactly one --> partition shape P_{s,c}
  +-- becomes exactly one --> X node X_{s,c}
```

For each retained point:

```text
point
  |
  +-- belongs to exactly one type
  +-- belongs to exactly one retained cluster
  +-- has one distance to its own cluster median
  +-- inherits centrality from its own X cluster node
  +-- receives one final SIGMA score
```

For each IO-DAG edge:

```text
a -> b
```

the edge:
1. survived MWAS using transaction value \(z_{ab}\);
2. is reweighted after MWAS with \(A_{ab}\);
3. enables X edges only where source and target cluster partition shapes overlap with positive area.

# 5. Intended primary output artifacts

## `sigma_clustered_points.parquet`
Step-2 result.

## `sigma_network_centers.parquet`
Step-3 result. One network 1-median per retained `[type, cluster]`.

## `sigma_points_with_center_distance.parquet`
Step-3 result. Retained points plus distance to their own cluster 1-median.

## `sigma_partitions.parquet`
Step-4 result. Exactly one Polygon/MultiPolygon per retained `[type, cluster]`.

## `sigma_io_dag.csv`
Step-5 result. MWAS edge selection uses transactions; final downstream edge weights are technical coefficients.

## `sigma_X_nodes.csv`
Step-6 result. One row per retained cluster node, including centrality.

## `sigma_X_edges.csv`
Step-6 result. Directed cluster-to-cluster spatial-economic edges.

## `sigma_disconnected_overlap_pairs.csv`
Step-6 audit result.

## `sigma_points_centrality.parquet`
Step-7 final point-level result.

# 6. Core implementation invariants

## Cluster-count invariants
If the retained data contain \(K\) clusters, then normally:

\[
K
=
n_{\text{network centers}}
=
n_{\text{partitions}}
=
n_{\text{X nodes}}.
\]

Any difference must be caused by an explicitly recorded recoverable failure; silent disagreement is an error.

## Partition invariants
Within each type:
- every retained cluster has one partition shape;
- every output geometry is Polygon/MultiPolygon;
- cluster identity is preserved;
- partition construction uses Step-2 clustered points, their cluster labels, road network, and study boundary.

## 1-median invariants
Every retained cluster has exactly one network 1-median.

Every retained point has one point-to-own-median road distance unless an explicitly handled connectivity failure occurs.

## MWAS invariants
MWAS selection weights are:

\[
z_{ab}.
\]

The MWAS result must be acyclic.

The final downstream DAG weight is:

\[
A_{ab}.
\]

## X-graph invariants
An X edge

\[
X_{a,i}\rightarrow X_{b,j}
\]

may exist only if:
1. \(a\rightarrow b\) survives MWAS;
2. \(P_{a,i}\) and \(P_{b,j}\) have positive-area overlap;
3. \(M_{a,i}\) and \(M_{b,j}\) are road-network connected.

Its stored quantities include:

\[
A_{ab},
\]

\[
d_G(M_{a,i},M_{b,j}),
\]

and:

\[
A_{ab}d_G(M_{a,i},M_{b,j}).
\]

## Point-score invariants
Each retained point's final score derives from:
- its own cluster's X centrality;
- its road distance to its own cluster median;
- the cluster-specific robust distance scale;
- configurable `distance_tempering`.

With default \(\lambda=0.15\):

\[
0.85C_{s,c}\le S_{s,c,i}\le C_{s,c}.
\]

# 7. Recommended defaults for the clean restart

```yaml
classification: io80

clustering:
  method: network_hdbscan
  min_cluster_size: 5

center:
  method: exact_network_1_median

partition:
  method: network_voronoi_from_clustered_points
  output_unit: type_cluster

io_dag:
  selection_matrix: transactions
  method: mwas_fast
  final_weight_matrix: technical_coefficients

x_graph:
  spatial_relation: positive_area_partition_overlap
  cluster_distance: road_distance_between_1_medians
  edge_weight: technical_coefficient_times_road_distance

point_score:
  distance_scale: cluster_p90
  distance_tempering: 0.15
  formula: cluster_centrality_times_bounded_distance_factor
```

# 8. Minimal mathematical specification

For type \(s\):

\[
C_{s,c}=\{p_{s,c,i}\}.
\]

Network median:

\[
M_{s,c}
=
\arg\min_v \sum_i d_G(v,p_{s,c,i}).
\]

Cluster shape:

\[
P_{s,c}
=
\text{network partition shape assigned to cluster }(s,c).
\]

MWAS:

\[
E_{\text{MWAS}}
=
\operatorname{MWAS}(Z),
\]

where selection weights are \(z_{ab}\).

Final DAG:

\[
G_D=(V,E_{\text{MWAS}},A_{ab}).
\]

Cluster-level edge existence:

\[
X_{a,i}\rightarrow X_{b,j}
\]

iff

\[
(a,b)\in E_{\text{MWAS}},
\]

\[
\operatorname{Area}(P_{a,i}\cap P_{b,j})>0,
\]

and

\[
d_G(M_{a,i},M_{b,j})<\infty.
\]

X-edge weight:

\[
w_{(a,i),(b,j)}
=
A_{ab}d_G(M_{a,i},M_{b,j}).
\]

Let cluster centrality be \(C^X_{s,c}\).

Cluster-specific distance scale:

\[
R_{s,c}
=
Q_{0.90}\{d_G(p_{s,c,i},M_{s,c})\}.
\]

Bounded normalized point distance:

\[
q_{s,c,i}
=
\min
\left(
\frac{d_G(p_{s,c,i},M_{s,c})}{\max(R_{s,c},1)},
1
\right).
\]

Final point score:

\[
\boxed{
S_{s,c,i}
=
C^X_{s,c}
\left(
1-0.15q_{s,c,i}
\right)
}
\]

by default.

# 9. Implementation principle for the restart

The clean `v0.1.0` implementation should mirror this dataflow directly:

```text
clustered_points
    |
    +--> centers + point_center_distances
    |
    +--> cluster_partitions

transactions --> MWAS --> surviving_edge_set
technical_coefficients --> reweight(surviving_edge_set)

cluster_partitions
+ centers
+ reweighted_DAG
+ roads
    |
    v
network_X
    |
    v
cluster_centrality

cluster_centrality
+ point_center_distances
    |
    v
point_scores
```

This dependency structure should be reflected consistently in:
- README;
- CLI terminology;
- checkpoints;
- metadata;
- artifact names;
- unit tests;
- integration tests.

# 10. Summary

SIGMA starts with classified points organized by IO type.

Network HDBSCAN produces clusters within each type.

Each retained cluster gets:
1. a network 1-median for distance calculations;
2. a polygonal partition shape for overlap testing;
3. one spatial-economic X node.

The IO transactions table determines the acyclic sector edge set through MWAS. The surviving DAG edges are then reweighted using technical coefficients.

A directed X edge is created when its type edge survives MWAS, the relevant cluster partition shapes overlap with positive area, and their cluster medians are road-network connected. Its current agreed weight is the technical coefficient multiplied by median-to-median road distance.

Centrality is computed at the cluster level.

Finally, each point inherits its cluster's centrality with a mild bounded distance adjustment. The recommended default distance-tempering coefficient is 0.15, limiting within-cluster attenuation to 15% so points within the same cluster remain substantially more alike than a raw inverse-distance formula would permit.
