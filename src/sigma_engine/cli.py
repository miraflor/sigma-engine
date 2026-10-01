"""Command-line interface for the focused SIGMA workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pipeline import EngineConfig, run_engine

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Build SIGMA's road-network/spatial economic centrality outputs.",
)


@app.callback()
def _root() -> None:
    """SIGMA Engine command group.

    Keeping an explicit callback is intentional.  Typer collapses an application that has
    only one command into a single root command; the callback preserves the documented
    ``sigma-engine run ...`` interface even while ``run`` is the only public subcommand.
    """


@app.command()
def run(
    roads: Annotated[
        Path,
        typer.Option("--roads", help="Projected road line dataset."),
    ],
    boundary: Annotated[
        Path,
        typer.Option("--boundary", help="Polygon boundary used to render partitions."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for final and audit outputs."),
    ],
    points: Annotated[
        Path,
        typer.Option(
            "--points",
            help=(
                "Existing classified point vector. Point production is external to "
                "sigma-engine."
            ),
        ),
    ],
    io_table_override: Annotated[
        Path | None,
        typer.Option(
            "--io-table-override",
            help=(
                "Optional custom IO transaction table. Normal runs use the bundled PSA "
                "2018 IO80/IO16 matrix selected by --classification."
            ),
        ),
    ] = None,
    classification: Annotated[
        str,
        typer.Option("--classification", help="Economic resolution: io80 (default) or io16."),
    ] = "io80",
    io80_column: Annotated[
        str | None,
        typer.Option(
            "--io80-column",
            help="Override IO80 field; default auto-detects io80_code or io80_map_code.",
        ),
    ] = None,
    io16_column: Annotated[
        str | None,
        typer.Option(
            "--io16-column",
            help="Override IO16 field; default auto-detects io16_code or io16_map_code.",
        ),
    ] = None,
    roads_layer: Annotated[
        str | None,
        typer.Option("--roads-layer", help="Layer name for a multi-layer road dataset."),
    ] = None,
    boundary_layer: Annotated[
        str | None,
        typer.Option("--boundary-layer", help="Layer name for a multi-layer boundary dataset."),
    ] = None,
    io_sheet: Annotated[
        str,
        typer.Option(
            "--io-sheet",
            help=(
                "Excel sheet for --io-table-override only; name or zero-based numeric index."
            ),
        ),
    ] = "0",
    mwas_method: Annotated[
        str,
        typer.Option(
            "--mwas-method",
            help=(
                "IO DAG transformation: fast (default deterministic heuristic) or "
                "exact (HiGHS MILP; may be much slower)."
            ),
        ),
    ] = "fast",
    min_cluster_size: Annotated[
        int,
        typer.Option("--min-cluster-size", help="HDBSCAN minimum cluster size (>=2)."),
    ] = 5,
    min_samples: Annotated[
        int | None,
        typer.Option("--min-samples", help="HDBSCAN min_samples; defaults to min cluster size."),
    ] = None,
    cluster_selection_method: Annotated[
        str,
        typer.Option(
            "--cluster-selection-method",
            help="HDBSCAN flat-cluster selection: eom or leaf.",
        ),
    ] = "eom",
    allow_single_cluster: Annotated[
        bool,
        typer.Option("--allow-single-cluster", help="Allow HDBSCAN to return one cluster."),
    ] = False,
    hdbscan_max_distance: Annotated[
        float,
        typer.Option(
            "--hdbscan-max-distance",
            help="Maximum sparse road-neighbour search radius; default 5000 road-CRS units.",
        ),
    ] = 5_000.0,
    hdbscan_distance_mode: Annotated[
        str,
        typer.Option(
            "--hdbscan-distance-mode",
            help="Sparse HDBSCAN distance search: adaptive (default) or fixed.",
        ),
    ] = "adaptive",
    hdbscan_min_distance: Annotated[
        float | None,
        typer.Option(
            "--hdbscan-min-distance",
            help="Optional first search radius in adaptive mode.",
        ),
    ] = None,
    hdbscan_max_neighbor_pairs: Annotated[
        int,
        typer.Option(
            "--hdbscan-max-neighbor-pairs",
            help="Safety cap on stored sparse neighbour pairs.",
        ),
    ] = 20_000_000,
    max_snap_distance: Annotated[
        float | None,
        typer.Option(
            "--max-snap-distance",
            help="Reject points farther than this from roads (road-CRS units).",
        ),
    ] = None,
    vertex_digits: Annotated[
        int,
        typer.Option(
            "--vertex-digits",
            help="Significant digits used to canonicalize road vertices.",
        ),
    ] = 11,
    voronoi_resolution: Annotated[
        float,
        typer.Option(
            "--voronoi-resolution",
            help="2-D Voronoi grid resolution in road-CRS units.",
        ),
    ] = 500.0,
    voronoi_max_cells: Annotated[
        int,
        typer.Option(
            "--voronoi-max-cells",
            help="Safety cap on candidate Voronoi surface grid cells.",
        ),
    ] = 1_000_000,
    voronoi_refine_factor: Annotated[
        float,
        typer.Option(
            "--voronoi-refine-factor",
            help="Multiply Voronoi resolution by this value when a grid is too coarse.",
        ),
    ] = 0.5,
    voronoi_max_refinements: Annotated[
        int,
        typer.Option(
            "--voronoi-max-refinements",
            help=(
                "Maximum network-Voronoi refinements after the initial attempt; "
                "default 5 = 6 total attempts before Euclidean fallback."
            ),
        ),
    ] = 5,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast",
            help="Abort on the first recoverable per-type/per-cluster error instead of skipping it.",
        ),
    ] = False,
    centrality_direction: Annotated[
        str,
        typer.Option(
            "--centrality-direction",
            help="Directed prestige convention: incoming (default) or outgoing.",
        ),
    ] = "incoming",
    zero_distance_floor: Annotated[
        float,
        typer.Option(
            "--zero-distance-floor",
            help="Finite denominator used only when a point is exactly at its center.",
        ),
    ] = 1.0,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Show live timestamped progress for long-running stages.",
        ),
    ] = True,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse matching durable stage checkpoints from this output directory.",
        ),
    ] = True,
    restart: Annotated[
        bool,
        typer.Option(
            "--restart",
            help="Discard prior SIGMA checkpoints/artifacts in the output directory and recompute.",
        ),
    ] = False,
) -> None:
    """Run the complete SIGMA engine pipeline."""
    if classification not in {"io80", "io16"}:
        raise typer.BadParameter("--classification must be io80 or io16")
    if cluster_selection_method not in {"eom", "leaf"}:
        raise typer.BadParameter("--cluster-selection-method must be eom or leaf")
    if centrality_direction not in {"incoming", "outgoing"}:
        raise typer.BadParameter("--centrality-direction must be incoming or outgoing")
    if hdbscan_distance_mode not in {"adaptive", "fixed"}:
        raise typer.BadParameter("--hdbscan-distance-mode must be adaptive or fixed")
    if mwas_method not in {"fast", "exact"}:
        raise typer.BadParameter("--mwas-method must be fast or exact")

    # Typer receives this option as text so users can specify either ``--io-sheet 0`` or a
    # literal worksheet name.  A purely decimal token is interpreted as a zero-based index.
    parsed_sheet: str | int
    try:
        parsed_sheet = int(io_sheet)
    except ValueError:
        parsed_sheet = io_sheet

    config = EngineConfig(
        roads_path=str(roads),
        boundary_path=str(boundary),
        io_table_override_path=str(io_table_override) if io_table_override else None,
        output_dir=str(output_dir),
        points_path=str(points),
        classification=classification,  # validated immediately above
        io80_column=io80_column,
        io16_column=io16_column,
        roads_layer=roads_layer,
        boundary_layer=boundary_layer,
        io_sheet=parsed_sheet,
        mwas_method=mwas_method,  # validated above
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,  # validated above
        allow_single_cluster=allow_single_cluster,
        hdbscan_max_distance=hdbscan_max_distance,
        hdbscan_distance_mode=hdbscan_distance_mode,
        hdbscan_min_distance=hdbscan_min_distance,
        hdbscan_max_neighbor_pairs=hdbscan_max_neighbor_pairs,
        max_snap_distance=max_snap_distance,
        vertex_digits=vertex_digits,
        voronoi_resolution=voronoi_resolution,
        voronoi_max_cells=voronoi_max_cells,
        voronoi_refine_factor=voronoi_refine_factor,
        voronoi_max_refinements=voronoi_max_refinements,
        fail_fast=fail_fast,
        centrality_direction=centrality_direction,  # validated above
        zero_distance_floor=zero_distance_floor,
        progress=progress,
        resume=resume,
        restart=restart,
    )

    try:
        result = run_engine(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"retained points: {len(result.points):,}")
    typer.echo(f"clusters: {len(result.centers):,}")
    typer.echo(f"partitions: {len(result.partitions):,}")
    for name, path in result.output_paths.items():
        typer.echo(f"{name}: {path}")


def main() -> None:
    """Console-script entry point."""
    app()
