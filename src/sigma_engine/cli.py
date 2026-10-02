"""Command-line interface for the revised SIGMA v0.1.0 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pipeline import ContinueConfig, EngineConfig, run_engine, run_from_partitions

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SIGMA: network clusters -> medians/partitions -> IO-DAG -> X -> point scores.",
)


def _sheet(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        return value


def _emit_result(result) -> None:
    typer.echo(f"points scored: {len(result.points):,}")
    typer.echo(f"clusters / X nodes: {len(result.centers):,}")
    typer.echo(f"partitions: {len(result.partitions):,}")
    for name, path in result.output_paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def run(
    points: Annotated[Path, typer.Option("--points", help="Classified point vector/GeoParquet.")],
    roads: Annotated[Path, typer.Option("--roads", help="Projected road line dataset.")],
    boundary: Annotated[Path, typer.Option("--boundary", help="Study boundary polygon.")],
    technical_coefficients: Annotated[
        Path,
        typer.Option(
            "--technical-coefficients",
            help="IO technical-coefficient matrix A. Required; never inferred from Z.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir", help="Output directory.")],
    transactions: Annotated[
        Path | None,
        typer.Option(
            "--transactions",
            help="Optional transaction matrix Z override; bundled PSA matrix is default.",
        ),
    ] = None,
    classification: Annotated[
        str, typer.Option("--classification", help="io80 (default) or io16.")
    ] = "io80",
    io80_column: Annotated[
        str | None, typer.Option("--io80-column", help="Override IO80 classification field.")
    ] = None,
    io16_column: Annotated[
        str | None, typer.Option("--io16-column", help="Override IO16 classification field.")
    ] = None,
    roads_layer: Annotated[
        str | None, typer.Option("--roads-layer", help="Road layer for multi-layer input.")
    ] = None,
    boundary_layer: Annotated[
        str | None, typer.Option("--boundary-layer", help="Boundary layer for multi-layer input.")
    ] = None,
    transactions_sheet: Annotated[
        str, typer.Option("--transactions-sheet", help="Worksheet name or zero-based index.")
    ] = "0",
    technical_coefficients_sheet: Annotated[
        str,
        typer.Option("--technical-coefficients-sheet", help="Worksheet name or zero-based index."),
    ] = "0",
    mwas_method: Annotated[
        str,
        typer.Option("--mwas-method", help="fast (default) or exact."),
    ] = "fast",
    min_cluster_size: Annotated[
        int, typer.Option("--min-cluster-size", help="Network HDBSCAN minimum cluster size.")
    ] = 5,
    min_samples: Annotated[
        int | None, typer.Option("--min-samples", help="HDBSCAN min_samples; defaults internally.")
    ] = None,
    cluster_selection_method: Annotated[
        str, typer.Option("--cluster-selection-method", help="eom or leaf.")
    ] = "eom",
    allow_single_cluster: Annotated[
        bool, typer.Option("--allow-single-cluster", help="Allow one HDBSCAN cluster per type.")
    ] = False,
    hdbscan_max_distance: Annotated[
        float, typer.Option("--hdbscan-max-distance", help="Maximum road-neighbour radius.")
    ] = 5_000.0,
    hdbscan_distance_mode: Annotated[
        str, typer.Option("--hdbscan-distance-mode", help="adaptive (default) or fixed.")
    ] = "adaptive",
    hdbscan_min_distance: Annotated[
        float | None, typer.Option("--hdbscan-min-distance", help="First adaptive search radius.")
    ] = None,
    hdbscan_max_neighbor_pairs: Annotated[
        int, typer.Option("--hdbscan-max-neighbor-pairs", help="Sparse pair safety cap.")
    ] = 20_000_000,
    max_snap_distance: Annotated[
        float | None, typer.Option("--max-snap-distance", help="Optional point-to-road QA limit.")
    ] = None,
    vertex_digits: Annotated[
        int, typer.Option("--vertex-digits", help="Road vertex canonicalization digits.")
    ] = 11,
    voronoi_resolution: Annotated[
        float,
        typer.Option(
            "--voronoi-resolution",
            help="Initial network-Voronoi surface resolution.",
        ),
    ] = 500.0,
    voronoi_max_cells: Annotated[
        int, typer.Option("--voronoi-max-cells", help="Network-Voronoi grid-cell safety cap.")
    ] = 1_000_000,
    voronoi_refine_factor: Annotated[
        float, typer.Option("--voronoi-refine-factor", help="Resolution multiplier on refinement.")
    ] = 0.5,
    voronoi_max_refinements: Annotated[
        int, typer.Option("--voronoi-max-refinements", help="Maximum network-only refinements.")
    ] = 5,
    centrality_direction: Annotated[
        str, typer.Option("--centrality-direction", help="incoming (default) or outgoing.")
    ] = "incoming",
    distance_tempering: Annotated[
        float,
        typer.Option(
            "--distance-tempering",
            help="Point attenuation lambda; default 0.15, bounded by cluster p90.",
        ),
    ] = 0.15,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show stage progress.")
    ] = True,
) -> None:
    """Run the complete revised SIGMA workflow from classified points."""
    config = EngineConfig(
        roads_path=str(roads),
        boundary_path=str(boundary),
        output_dir=str(output_dir),
        points_path=str(points),
        technical_coefficients_path=str(technical_coefficients),
        classification=classification,  # validated in pipeline
        transactions_override_path=str(transactions) if transactions else None,
        io80_column=io80_column,
        io16_column=io16_column,
        roads_layer=roads_layer,
        boundary_layer=boundary_layer,
        transactions_sheet=_sheet(transactions_sheet),
        technical_coefficients_sheet=_sheet(technical_coefficients_sheet),
        mwas_method=mwas_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
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
        centrality_direction=centrality_direction,
        distance_tempering=distance_tempering,
        progress=progress,
    )
    try:
        result = run_engine(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_result(result)


@app.command("from-partitions")
def from_partitions(
    partitions: Annotated[
        Path,
        typer.Option(
            "--partitions",
            help="Existing sigma_partitions.parquet from Step 4.",
        ),
    ],
    roads: Annotated[Path, typer.Option("--roads", help="Same projected road dataset.")],
    technical_coefficients: Annotated[
        Path,
        typer.Option("--technical-coefficients", help="IO technical-coefficient matrix A."),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Output directory for Steps 5-7.")
    ],
    centers: Annotated[
        Path | None,
        typer.Option(
            "--centers",
            help="Step-3 centers; defaults to sigma_network_centers.parquet beside partitions.",
        ),
    ] = None,
    points_with_center_distance: Annotated[
        Path | None,
        typer.Option(
            "--points-with-center-distance",
            help=(
                "Step-3 point-distance artifact; defaults to "
                "sigma_points_with_center_distance.parquet beside partitions."
            ),
        ),
    ] = None,
    transactions: Annotated[
        Path | None,
        typer.Option("--transactions", help="Optional transaction matrix Z override."),
    ] = None,
    classification: Annotated[
        str, typer.Option("--classification", help="io80 (default) or io16.")
    ] = "io80",
    roads_layer: Annotated[
        str | None, typer.Option("--roads-layer", help="Road layer for multi-layer input.")
    ] = None,
    transactions_sheet: Annotated[
        str, typer.Option("--transactions-sheet", help="Worksheet name or zero-based index.")
    ] = "0",
    technical_coefficients_sheet: Annotated[
        str,
        typer.Option("--technical-coefficients-sheet", help="Worksheet name or zero-based index."),
    ] = "0",
    mwas_method: Annotated[
        str, typer.Option("--mwas-method", help="fast (default) or exact.")
    ] = "fast",
    vertex_digits: Annotated[
        int, typer.Option("--vertex-digits", help="Road vertex canonicalization digits.")
    ] = 11,
    centrality_direction: Annotated[
        str, typer.Option("--centrality-direction", help="incoming (default) or outgoing.")
    ] = "incoming",
    distance_tempering: Annotated[
        float,
        typer.Option("--distance-tempering", help="Point attenuation lambda; default 0.15."),
    ] = 0.15,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show stage progress.")
    ] = True,
) -> None:
    """Run only Steps 5--7 from existing Voronoi/median artifacts."""
    config = ContinueConfig(
        roads_path=str(roads),
        partitions_path=str(partitions),
        technical_coefficients_path=str(technical_coefficients),
        output_dir=str(output_dir),
        centers_path=str(centers) if centers else None,
        points_with_center_distance_path=(
            str(points_with_center_distance) if points_with_center_distance else None
        ),
        classification=classification,
        transactions_override_path=str(transactions) if transactions else None,
        roads_layer=roads_layer,
        transactions_sheet=_sheet(transactions_sheet),
        technical_coefficients_sheet=_sheet(technical_coefficients_sheet),
        mwas_method=mwas_method,
        vertex_digits=vertex_digits,
        centrality_direction=centrality_direction,
        distance_tempering=distance_tempering,
        progress=progress,
    )
    try:
        result = run_from_partitions(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_result(result)


def main() -> None:
    app()
