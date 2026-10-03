# Changelog

## 0.1.0 — revised workflow reset

- Reset package version to `0.1.0` for the clean workflow restart.
- Separate IO edge **selection** (`Z`, transactions) from downstream edge **weighting** (`A`, technical coefficients).
- Keep fast MWAS as the default and exact MWAS as opt-in.
- Enforce one retained cluster = one exact network median = one partition = one X node.
- Remove Euclidean center/Voronoi substitution from the revised full run.
- Keep positive-area overlap as the only spatial relation that instantiates an IO-DAG edge in X.
- Keep X edge weight `technical_coefficient * median_to_median_road_distance`.
- Replace inverse-distance point scoring with cluster-p90 bounded tempering; default `lambda=0.15`.
- Add `sigma-engine from-partitions` so existing Step-3/4 artifacts can be tested immediately.
- Rebuild center road nodes from saved median geometries on restart instead of trusting old augmented node IDs.
- Align public artifact names and columns with the revised workflow specification.

## Overlay correction r2

- Preserve normalized sector codes such as `01` when restoring legacy X CSV artifacts.
- Make CLI option tests independent of Rich/Typer terminal-width wrapping.
- Reject legacy Euclidean point-to-center fallback distances in `from-partitions`.
- Validate restart CRS, point geometry, non-negative clusters/distances, and unique point IDs.
- Normalize type identifiers consistently across saved center/partition/restart artifacts.
- Include the Step-1 snapped network position and snap distance in the public Step-2 clustered artifact.
- Fix overlay-owned Ruff findings and replace the misleading full-repository Ruff gate with a scoped lint gate.

## Overlay correction r4

- Make the verifier normalize imports with Ruff's own `I`-rule fixer before the lint gate.
- Immediately re-run Ruff without fixes so import normalization cannot hide a remaining lint failure.
- Verify the downloaded overlay ZIP checksum before extraction.
- No runtime, algorithm, CLI, artifact-schema, or scoring behavior changes from r2/r3.
- Continue to run the full repository pytest suite as the behavioral acceptance gate.

## Built-in PSA economic-input patch r6

- Bundle compact PSA 2018 gross-output vectors for both IO80 and IO16 beside the existing transaction matrices.
- Make normal IO80/IO16 runs fully self-contained: bundled `Z` selects MWAS edges and bundled `x` derives `A_ij = Z_ij / x_j`.
- Keep `--transactions` and `--technical-coefficients` only as explicit custom overrides.
- Verify all four bundled economic resources by SHA-256 at load time.
- Record the built-in gross-output resource hash in run metadata.
- Preserve full-workbook Total Output extraction for custom transaction overrides.
- Add regression tests proving that no external economic file is required and that intermediate column sums are never used as gross output.

## Centrality correction r7b

- Preserve directed X edges for audit, but remove directionality before eigenvector centrality.
- Compute weighted eigenvector centrality on the positive-weight undirected projection of X.
- Keep the legacy centrality-direction option temporarily for compatibility; it no longer changes the result.
- Mark run metadata as `undirected_weighted_eigenvector` / `undirected`.
- Remove the DAG terminal-node fallback that concentrated centrality on sinks.

## Spatial outputs + sigma-siphon area integration overlay r8

- Add one `sigma_spatial_outputs.gpkg` per run with `clusters`, `nodes`, and `paths` layers.
- Preserve exact Dijkstra predecessor paths so every X edge can be exported as its actual road-network route.
- Add node latitude/longitude in EPSG:4326 plus directed in-degree and out-degree.
- Add `OUTPUTS.txt` with stable artifact descriptions and run-specific network/path statistics.
- Add `sigma-engine area <area>` to resolve sigma-siphon `areas.yml`, reuse valid existing POIs, or invoke sigma-siphon when they are absent.
- Materialize only the selected area boundary from sigma-siphon's national boundary GeoPackage.
- Keep the road dataset explicit: sigma-siphon currently does not publish the routable road network required by SIGMA Engine.
