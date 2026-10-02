from types import SimpleNamespace

from typer.main import get_command
from typer.testing import CliRunner

from sigma_engine.cli import app


def _result():
    return SimpleNamespace(points=[], centers=[], partitions=[], output_paths={})


def test_cli_exposes_full_and_restart_commands():
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "run" in root.stdout
    assert "from-partitions" in root.stdout

    full = runner.invoke(app, ["run", "--help"])
    assert full.exit_code == 0
    restart = runner.invoke(app, ["from-partitions", "--help"])
    assert restart.exit_code == 0

    # Inspect Click's option declarations instead of Rich-rendered help text.  Typer can
    # wrap long option names differently depending on terminal width and version.
    root_command = get_command(app)
    run_options = {
        option
        for parameter in root_command.commands["run"].params
        for option in getattr(parameter, "opts", [])
    }
    restart_options = {
        option
        for parameter in root_command.commands["from-partitions"].params
        for option in getattr(parameter, "opts", [])
    }
    assert {
        "--technical-coefficients",
        "--transactions",
        "--mwas-method",
        "--distance-tempering",
    } <= run_options
    assert {
        "--partitions",
        "--centers",
        "--points-with-center-distance",
    } <= restart_options


def test_cli_fast_mwas_and_tempering_are_defaults(monkeypatch):
    captured = []

    def fake(config):
        captured.append(config)
        return _result()

    monkeypatch.setattr("sigma_engine.cli.run_engine", fake)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--points", "points.parquet",
            "--roads", "roads.gpkg",
            "--boundary", "boundary.gpkg",
            "--output-dir", "out",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured[-1].classification == "io80"
    assert captured[-1].mwas_method == "fast"
    assert captured[-1].distance_tempering == 0.15
    assert captured[-1].transactions_override_path is None
    assert captured[-1].technical_coefficients_path is None


def test_from_partitions_passes_restart_artifacts(monkeypatch):
    captured = []

    def fake(config):
        captured.append(config)
        return _result()

    monkeypatch.setattr("sigma_engine.cli.run_from_partitions", fake)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "from-partitions",
            "--partitions", "sigma_partitions.parquet",
            "--roads", "roads.gpkg",
            "--output-dir", "out",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured[-1].centers_path is None
    assert captured[-1].points_with_center_distance_path is None
    assert captured[-1].mwas_method == "fast"
    assert captured[-1].technical_coefficients_path is None
