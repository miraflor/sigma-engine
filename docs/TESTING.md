# Testing the revised overlay

After copying the overlay onto the current repository:

```powershell
cd "C:\Users\James Matthew\Documents\GitHub\sigma-engine"
python -m pip install -e ".[dev]"
python -m compileall -q src tests
python -m ruff check src/sigma_engine/__init__.py src/sigma_engine/__main__.py src/sigma_engine/_version.py src/sigma_engine/cli.py src/sigma_engine/io_workflow.py src/sigma_engine/pipeline.py src/sigma_engine/scoring.py tests/test_cli.py tests/test_io_workflow_revised.py tests/test_pipeline.py tests/test_pipeline_contracts.py tests/test_scoring.py
python -m pytest -q
python -m sigma_engine --help
python -m sigma_engine from-partitions --help
```

## Immediate test using already-computed partitions

If the three Step-3/4 artifacts are in the same existing output directory:

```text
sigma_network_centers.parquet
sigma_points_with_center_distance.parquet
sigma_partitions.parquet
```

run:

```powershell
python -m sigma_engine from-partitions `
  --partitions "C:\path\existing_output\sigma_partitions.parquet" `
  --roads "C:\path\same_roads.gpkg" `
  --output-dir "C:\path\revised_output"
```

The command auto-detects the center and point-distance files beside the partition file. It uses bundled PSA IO80 by default (or IO16 when selected), derives the technical-coefficient matrix from the bundled gross-output vector, and does not rerun HDBSCAN, 1-medians, or network Voronoi. `--transactions` and `--technical-coefficients` remain explicit custom overrides only.

Expected revised outputs:

```text
sigma_io_dag.csv
sigma_X_nodes.csv
sigma_X_edges.csv
sigma_disconnected_overlap_pairs.csv
sigma_points_centrality.parquet
sigma_run_metadata.json
```

Check the key invariants quickly:

```powershell
python -c "import pandas as pd, geopandas as gpd; p=gpd.read_parquet(r'C:\path\existing_output\sigma_partitions.parquet'); c=gpd.read_parquet(r'C:\path\existing_output\sigma_network_centers.parquet'); x=pd.read_csv(r'C:\path\revised_output\sigma_X_nodes.csv'); print('partitions',len(p),'centers',len(c),'X nodes',len(x)); assert len(p)==len(c)==len(x)"
```

And verify the final bounded point score:

```powershell
python -c "import geopandas as gpd, numpy as np; g=gpd.read_parquet(r'C:\path\revised_output\sigma_points_centrality.parquet'); lo=g.cluster_centrality*(1-g.distance_tempering); assert np.all(g.sigma_score>=lo-1e-12); assert np.all(g.sigma_score<=g.cluster_centrality+1e-12); print(g[['sigma_score','cluster_centrality','normalized_cluster_distance']].describe())"
```

## Ruff scope

The inspected baseline commit already contains Ruff findings in unchanged spatial modules
(`_sparse_hdbscan.py`, `_sparse_network.py`, `network.py`, `voronoi.py`, and others).
Those are outside this workflow overlay. Therefore the overlay acceptance check lints the
files changed or added by this reset, while the full pytest suite remains repository-wide.
A separate lint-only cleanup can be done later without mixing formatting edits into the
workflow correction.
