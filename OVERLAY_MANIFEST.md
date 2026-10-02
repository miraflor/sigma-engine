# Overlay manifest

Prepared against `miraflor/sigma-engine` `main` at commit
`b02185c2f2e821348e1b03bbcf2cb3640e0588e0`.

## Files replaced

```text
README.md
APPLY.txt
pyproject.toml
src/sigma_engine/__init__.py
src/sigma_engine/__main__.py
src/sigma_engine/_version.py
src/sigma_engine/cli.py
src/sigma_engine/pipeline.py
tests/test_cli.py
tests/test_pipeline.py
tests/test_pipeline_contracts.py
```

## Files added

```text
CHANGELOG.md
OVERLAY_MANIFEST.md
docs/SIGMA_REVISED_WORKFLOW.md
docs/IMPLEMENTATION_NOTES.md
src/sigma_engine/io_workflow.py
src/sigma_engine/scoring.py
tests/test_io_workflow_revised.py
tests/test_scoring.py
```

## Existing files intentionally retained unchanged

The overlay expects the current repository versions of:

```text
src/sigma_engine/_sparse_hdbscan.py
src/sigma_engine/_sparse_network.py
src/sigma_engine/_sparse_solver.py
src/sigma_engine/builtin_io.py
src/sigma_engine/center.py
src/sigma_engine/clustering.py
src/sigma_engine/io_dag.py
src/sigma_engine/io_utils.py
src/sigma_engine/network.py
src/sigma_engine/point_input.py
src/sigma_engine/progress.py
src/sigma_engine/spatial_graph.py
src/sigma_engine/voronoi.py
src/sigma_engine/resources/*
```

This is deliberate: the revised workflow changes their composition and contracts, not
the already-working spatial/MWAS primitives themselves.

## Stale files that may be deleted

```text
APPLY_AUGMENTED_ROAD_CHECKPOINT.txt
APPLY_XFIX.txt
```

The new runtime ignores old `.sigma_checkpoints` directories. Existing output artifacts
remain usable through `sigma-engine from-partitions`.

## Correction r2

This package supersedes the first v0.1.0 overlay produced on 2026-10-02. It keeps the
same target reset version and baseline commit, but fixes the first overlay's acceptance
and compatibility defects:

```text
- legacy CSV sector IDs are normalized on restore (01 stays 01)
- CLI tests inspect declared options rather than terminal-wrapped Rich help
- restart artifacts reject legacy Euclidean point-center fallbacks
- restart artifact CRS/geometry/distance/point-ID invariants are validated
- Step-2 public clustered output carries snapped network position and snap distance
- overlay-owned lint findings are fixed
- fail-fast application/verification script is included under tools/
```

The repository-wide Ruff findings in untouched baseline spatial modules are not modified
by this workflow overlay. Full repository pytest remains the behavioral acceptance gate.

## Correction r4

This package supersedes r2 only to correct the remaining Ruff I001 finding in
`src/sigma_engine/pipeline.py`. The change is import formatting only; workflow semantics
and all public contracts are unchanged from r2.

## r4 verification note

The r4 application script intentionally runs `ruff check --select I --fix` on the
overlay-owned Python files before the ordinary Ruff gate. This delegates import ordering
to the exact Ruff version installed in the target environment, then verifies the resulting
files with a second no-fix lint pass. The import-only normalization does not change runtime
logic.
