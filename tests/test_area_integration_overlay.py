from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import box

from sigma_engine.area_integration import (
    ensure_siphon_output,
    load_area_definition,
    materialize_area_boundary,
)


def test_area_definition_reuses_existing_siphon_output_and_materializes_boundary(
    tmp_path: Path, monkeypatch
):
    siphon = tmp_path / "sigma-siphon"
    (siphon / "config").mkdir(parents=True)
    (siphon / "src" / "sigma_siphon").mkdir(parents=True)
    (siphon / "src" / "sigma_siphon" / "pipeline.py").write_text("# test\n")
    (siphon / "data" / "boundaries").mkdir(parents=True)

    boundary_gpkg = siphon / "data" / "boundaries" / "areas.gpkg"
    gpd.GeoDataFrame(
        {"psgc_code": ["0123456789", "9999999999"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    ).to_file(boundary_gpkg, layer="areas", driver="GPKG")

    areas_file = siphon / "config" / "areas.yml"
    areas_file.write_text(
        yaml.safe_dump(
            {
                "areas": {
                    "sample": {
                        "name": "Sample",
                        "kind": "city",
                        "aliases": ["0123456789"],
                        "bbox": [0, 0, 1, 1],
                        "boundary": {
                            "gpkg": "../data/boundaries/areas.gpkg",
                            "layer": "areas",
                            "field": "psgc_code",
                            "value": "0123456789",
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    area = load_area_definition("0123456789", areas_file)
    assert area.slug == "sample"

    output_root = siphon / "output"
    area_output = output_root / "sample"
    area_output.mkdir(parents=True)
    points_file = area_output / "pois.parquet"
    points_file.write_bytes(b"test-placeholder")
    monkeypatch.setattr(
        "sigma_engine.area_integration._validate_siphon_points",
        lambda path, classification: None,
    )

    points, run_json, was_run = ensure_siphon_output(
        area,
        siphon_root=siphon,
        areas_file=areas_file,
        classification="io80",
        output_root=output_root,
    )
    assert points == area_output / "pois.parquet"
    assert run_json is None
    assert was_run is False

    boundary = materialize_area_boundary(area, tmp_path / "selected.gpkg")
    selected = gpd.read_file(boundary, layer="boundary")
    assert len(selected) == 1
    assert selected.loc[0, "area_slug"] == "sample"
    assert selected.geometry.iloc[0].equals(box(0, 0, 1, 1))


def test_cli_exposes_area_command():
    from typer.testing import CliRunner

    from sigma_engine.cli import app

    result = CliRunner().invoke(app, ["area", "--help"])
    assert result.exit_code == 0
    assert "--siphon-root" in result.stdout
    assert "--refresh-siphon" in result.stdout
    assert "--roads" in result.stdout
