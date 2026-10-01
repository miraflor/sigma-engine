from types import SimpleNamespace

from typer.testing import CliRunner

from sigma_engine.cli import app


def test_cli_keeps_explicit_run_subcommand():
    runner = CliRunner()
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "run" in root.stdout

    command = runner.invoke(app, ["run", "--help"])
    assert command.exit_code == 0
    assert "--points" in command.stdout
    assert "--roads" in command.stdout
    assert "--io-table-override" in command.stdout
    assert "--resume" in command.stdout
    assert "--no-resume" in command.stdout
    assert "--restart" in command.stdout
    assert "--mwas-method" in command.stdout
    assert "--area" not in command.stdout
    assert "--siphon" not in command.stdout


def test_cli_passes_existing_points_without_any_producer_options(monkeypatch):
    captured = []

    def fake_run_engine(config):
        captured.append(config)
        return SimpleNamespace(points=[], centers=[], partitions=[], output_paths={})

    monkeypatch.setattr("sigma_engine.cli.run_engine", fake_run_engine)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--points",
            "pois.parquet",
            "--roads",
            "roads.gpkg",
            "--boundary",
            "boundary.gpkg",
            "--output-dir",
            "output",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured[-1].points_path == "pois.parquet"
    assert captured[-1].io80_column is None
    assert captured[-1].io16_column is None
    assert captured[-1].io_table_override_path is None
    assert captured[-1].mwas_method == "fast"


def test_cli_allows_explicit_exact_mwas(monkeypatch):
    captured = []

    def fake_run_engine(config):
        captured.append(config)
        return SimpleNamespace(points=[], centers=[], partitions=[], output_paths={})

    monkeypatch.setattr("sigma_engine.cli.run_engine", fake_run_engine)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--points", "pois.parquet",
            "--roads", "roads.gpkg",
            "--boundary", "boundary.gpkg",
            "--output-dir", "output",
            "--mwas-method", "exact",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured[-1].mwas_method == "exact"
