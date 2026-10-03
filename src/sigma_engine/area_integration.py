"""Area-oriented handoff from sigma-siphon into sigma-engine.

The integration deliberately treats sigma-siphon as a separate application.  SIGMA Engine
reads its public area catalog and output files, and launches its public pipeline in a
separate Python process only when a usable area output is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import geopandas as gpd
import yaml
from shapely.geometry import box


@dataclass(frozen=True)
class AreaDefinition:
    slug: str
    name: str
    kind: str
    bbox: tuple[float, float, float, float]
    boundary_gpkg: Path | None
    boundary_layer: str | None
    boundary_field: str | None
    boundary_values: tuple[str, ...]


@dataclass(frozen=True)
class AreaInputs:
    area: AreaDefinition
    points_path: Path
    boundary_path: Path
    boundary_layer: str
    siphon_run_json: Path | None
    siphon_was_run: bool
    siphon_root: Path
    areas_file: Path


def _is_siphon_root(path: Path) -> bool:
    return (
        (path / "config" / "areas.yml").is_file()
        and (path / "src" / "sigma_siphon" / "pipeline.py").is_file()
    )


def resolve_siphon_root(explicit: str | Path | None = None) -> Path:
    """Resolve a sigma-siphon checkout without importing it into sigma-engine."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("SIGMA_SIPHON_ROOT")
    if env:
        candidates.append(Path(env))
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "sigma-siphon",
            cwd / "sigma-siphon-main",
            cwd.parent / "sigma-siphon",
            cwd.parent / "sigma-siphon-main",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_siphon_root(resolved):
            return resolved
    searched = ", ".join(str(value) for value in seen) or "<no candidates>"
    raise FileNotFoundError(
        "could not locate a sigma-siphon checkout. Pass --siphon-root or set "
        f"SIGMA_SIPHON_ROOT. Searched: {searched}"
    )


def resolve_areas_file(siphon_root: Path, explicit: str | Path | None = None) -> Path:
    path = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else (siphon_root / "config" / "areas.yml").resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"sigma-siphon areas YAML does not exist: {path}")
    return path


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def load_area_definition(value: str, areas_file: str | Path) -> AreaDefinition:
    """Resolve an area slug/name/alias from sigma-siphon's YAML contract."""
    path = Path(areas_file).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    areas = payload.get("areas")
    if not isinstance(areas, dict):
        raise ValueError(f"{path}: expected an 'areas' mapping")

    query = str(value).casefold().strip()
    matches: list[tuple[str, dict[str, object]]] = []
    for slug, raw in areas.items():
        if not isinstance(raw, dict):
            continue
        candidates = {str(slug).casefold(), str(raw.get("name") or slug).casefold()}
        candidates.update(alias.casefold() for alias in _string_tuple(raw.get("aliases")))
        if query in candidates:
            matches.append((str(slug), raw))
    if not matches:
        raise KeyError(f"unknown sigma-siphon area {value!r} in {path}")
    if len(matches) > 1:
        raise KeyError(
            f"ambiguous sigma-siphon area {value!r}: {[slug for slug, _ in matches]}"
        )

    slug, raw = matches[0]
    bbox_raw = raw.get("bbox")
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError(f"area {slug!r} has no valid bbox")
    bbox = tuple(float(item) for item in bbox_raw)
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(f"area {slug!r} has an invalid bbox")

    boundary = raw.get("boundary")
    boundary_gpkg: Path | None = None
    boundary_layer: str | None = None
    boundary_field: str | None = None
    boundary_values: tuple[str, ...] = ()
    if boundary not in (None, ""):
        if not isinstance(boundary, dict) or not boundary.get("gpkg"):
            raise ValueError(f"area {slug!r} has an invalid boundary definition")
        boundary_gpkg = Path(str(boundary["gpkg"]))
        if not boundary_gpkg.is_absolute():
            boundary_gpkg = (path.parent / boundary_gpkg).resolve()
        boundary_layer = str(boundary["layer"]) if boundary.get("layer") else None
        boundary_field = str(boundary["field"]) if boundary.get("field") else None
        if boundary.get("values") is not None:
            boundary_values = _string_tuple(boundary.get("values"))
        elif boundary.get("value") is not None:
            boundary_values = _string_tuple(boundary.get("value"))

    return AreaDefinition(
        slug=slug,
        name=str(raw.get("name") or slug),
        kind=str(raw.get("kind") or "area"),
        bbox=(west, south, east, north),
        boundary_gpkg=boundary_gpkg,
        boundary_layer=boundary_layer,
        boundary_field=boundary_field,
        boundary_values=boundary_values,
    )


def _validate_siphon_points(path: Path, classification: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"sigma-siphon POI output does not exist: {path}")
    classification_column = "io80_code" if classification == "io80" else "io16_code"
    columns = ["canonical_id", classification_column, "geometry"]
    try:
        frame = gpd.read_parquet(path, columns=columns)
    except Exception as exc:  # corrupted/incompatible parquet should fail closed
        raise ValueError(f"could not read sigma-siphon POI output {path}: {exc}") from exc
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"sigma-siphon POI output is missing required columns: {sorted(missing)}"
        )
    if frame.crs is None:
        raise ValueError("sigma-siphon POI output is missing its CRS")
    if frame["canonical_id"].isna().any():
        raise ValueError("sigma-siphon POI output contains missing canonical_id values")
    classified = frame[classification_column].astype("string").fillna("").str.strip().ne("")
    if not bool(classified.any()):
        raise ValueError(
            f"sigma-siphon POI output contains no usable {classification} classifications"
        )


def _run_siphon_subprocess(
    *,
    area: AreaDefinition,
    siphon_root: Path,
    areas_file: Path,
    output_root: Path,
    cache_dir: Path | None,
    progress: Callable[[str], None] | None,
) -> None:
    """Launch sigma-siphon's public run_pipeline in a separate Python process.

    This bypasses the interactive Geofabrik refresh prompt while retaining sigma-siphon's
    own acquisition/cache logic.  If the current source is absent, sigma-siphon still
    acquires what its pipeline requires; this bridge merely declines a forced refresh.
    """
    script = r'''
import os
from pathlib import Path
from sigma_siphon.pipeline import run_pipeline

def report(message: str) -> None:
    print(f"[sigma-siphon] {message}", flush=True)

run_pipeline(
    os.environ["SIGMA_BRIDGE_AREA"],
    areas_file=Path(os.environ["SIGMA_BRIDGE_AREAS_FILE"]),
    root=Path(os.environ["SIGMA_BRIDGE_ROOT"]),
    output_dir=Path(os.environ["SIGMA_BRIDGE_OUTPUT_ROOT"]),
    cache_dir=(
        Path(os.environ["SIGMA_BRIDGE_CACHE_DIR"])
        if os.environ.get("SIGMA_BRIDGE_CACHE_DIR")
        else None
    ),
    refresh=False,
    refresh_geofabrik=False,
    clip=True,
    use_llm=False,
    progress=report,
)
'''
    env = os.environ.copy()
    source_path = str((siphon_root / "src").resolve())
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["SIGMA_BRIDGE_AREA"] = area.slug
    env["SIGMA_BRIDGE_AREAS_FILE"] = str(areas_file)
    env["SIGMA_BRIDGE_ROOT"] = str(siphon_root)
    env["SIGMA_BRIDGE_OUTPUT_ROOT"] = str(output_root)
    env["SIGMA_BRIDGE_CACHE_DIR"] = str(cache_dir.resolve()) if cache_dir is not None else ""
    if progress is not None:
        progress(f"sigma-siphon output absent; running area {area.slug!r}")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=siphon_root,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sigma-siphon failed for area {area.slug!r} with exit code {completed.returncode}"
        )


def ensure_siphon_output(
    area: AreaDefinition,
    *,
    siphon_root: Path,
    areas_file: Path,
    classification: str,
    output_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path | None, bool]:
    """Reuse a valid sigma-siphon area output or create it on demand."""
    resolved_output_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (siphon_root / "output").resolve()
    )
    area_output = resolved_output_root / area.slug
    points = area_output / "pois.parquet"
    run_json = area_output / "run.json"

    if not refresh and points.is_file():
        try:
            _validate_siphon_points(points, classification)
            if run_json.is_file():
                report = json.loads(run_json.read_text(encoding="utf-8"))
                report_slug = str((report.get("area") or {}).get("slug") or "")
                if report_slug and report_slug != area.slug:
                    raise ValueError(
                        f"sigma-siphon run.json area {report_slug!r} does not match {area.slug!r}"
                    )
            if progress is not None:
                progress(f"using existing sigma-siphon output: {points}")
            return points, run_json if run_json.is_file() else None, False
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            if progress is not None:
                progress(f"existing sigma-siphon output is unusable ({exc}); rebuilding")

    resolved_cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    _run_siphon_subprocess(
        area=area,
        siphon_root=siphon_root,
        areas_file=areas_file,
        output_root=resolved_output_root,
        cache_dir=resolved_cache,
        progress=progress,
    )
    _validate_siphon_points(points, classification)
    return points, run_json if run_json.is_file() else None, True


def materialize_area_boundary(
    area: AreaDefinition,
    target: str | Path,
) -> Path:
    """Write exactly one area boundary feature for sigma-engine consumption."""
    destination = Path(target).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if area.boundary_gpkg is None:
        west, south, east, north = area.bbox
        selected = gpd.GeoDataFrame(
            {"area_slug": [area.slug], "area_name": [area.name]},
            geometry=[box(west, south, east, north)],
            crs="EPSG:4326",
        )
    else:
        if not area.boundary_gpkg.is_file():
            raise FileNotFoundError(
                f"configured area boundary GeoPackage does not exist: {area.boundary_gpkg}"
            )
        frame = gpd.read_file(area.boundary_gpkg, layer=area.boundary_layer)
        if frame.crs is None:
            raise ValueError("configured area boundary is missing its CRS")
        if area.boundary_field:
            if area.boundary_field not in frame.columns:
                raise ValueError(
                    f"area boundary field {area.boundary_field!r} is absent from "
                    f"{area.boundary_gpkg}"
                )
            values = (
                frame[area.boundary_field]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            wanted = tuple(str(value).strip() for value in area.boundary_values)
            mask = values.isin(wanted)
            for item in wanted:
                if item.isdigit():
                    mask |= values.str.replace(r"^0+", "", regex=True) == item.lstrip("0")
            frame = frame.loc[mask].copy()
        if frame.empty:
            raise ValueError(f"area {area.slug!r} selects no configured boundary features")
        geometry = frame.geometry.union_all()
        selected = gpd.GeoDataFrame(
            {"area_slug": [area.slug], "area_name": [area.name]},
            geometry=[geometry],
            crs=frame.crs,
        )
    if destination.exists():
        destination.unlink()
    selected.to_file(destination, layer="boundary", driver="GPKG")
    return destination


def prepare_area_inputs(
    value: str,
    *,
    engine_output_dir: str | Path,
    classification: str,
    siphon_root: str | Path | None = None,
    areas_file: str | Path | None = None,
    siphon_output_root: str | Path | None = None,
    siphon_cache_dir: str | Path | None = None,
    refresh_siphon: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AreaInputs:
    root = resolve_siphon_root(siphon_root)
    area_file = resolve_areas_file(root, areas_file)
    area = load_area_definition(value, area_file)
    points, run_json, was_run = ensure_siphon_output(
        area,
        siphon_root=root,
        areas_file=area_file,
        classification=classification,
        output_root=siphon_output_root,
        cache_dir=siphon_cache_dir,
        refresh=refresh_siphon,
        progress=progress,
    )
    boundary = materialize_area_boundary(
        area,
        Path(engine_output_dir) / "_inputs" / f"{area.slug}_boundary.gpkg",
    )
    return AreaInputs(
        area=area,
        points_path=points,
        boundary_path=boundary,
        boundary_layer="boundary",
        siphon_run_json=run_json,
        siphon_was_run=was_run,
        siphon_root=root,
        areas_file=area_file,
    )
