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
