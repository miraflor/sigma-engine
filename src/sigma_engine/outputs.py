"""User-facing spatial outputs and run description for SIGMA Engine."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point

from .network import AugmentedNetwork
from .spatial_graph import make_node_id

WGS84 = "EPSG:4326"


def build_x_node_outputs(
    graph: nx.DiGraph,
    centrality: Mapping[str, float],
    crs: object,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Create tabular and WGS84 point representations of X nodes."""
    rows: list[dict[str, object]] = []
    geometries = []
    for node, data in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        node_id = str(node)
        rows.append(
            {
                "node_id": node_id,
                "type": str(data["type"]),
                "cluster": int(data["cluster"]),
                "cluster_no": int(data["cluster"]),
                "in_degree": int(graph.in_degree(node)),
                "out_degree": int(graph.out_degree(node)),
                "centrality": float(centrality[node_id]),
            }
        )
        geometries.append(Point(float(data["center_x"]), float(data["center_y"])))

    geo = gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)
    geo = geo.to_crs(WGS84)
    geo["lon"] = geo.geometry.x.astype(float)
    geo["lat"] = geo.geometry.y.astype(float)
    order = [
        "node_id",
        "type",
        "cluster",
        "cluster_no",
        "lat",
        "lon",
        "in_degree",
        "out_degree",
        "centrality",
        "geometry",
    ]
    geo = geo[order]
    table = pd.DataFrame(geo.drop(columns="geometry"))
    return table, geo


def build_x_edge_outputs(
    graph: nx.DiGraph,
    road: AugmentedNetwork,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Create tabular X edges and exact shortest-road-path line geometry."""
    rows: list[dict[str, object]] = []
    geometries = []
    for source, target, data in sorted(
        graph.edges(data=True), key=lambda edge: (str(edge[0]), str(edge[1]))
    ):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        road_path = data.get("road_path")
        if road_path is None:
            raise RuntimeError(f"X edge {source!r}->{target!r} is missing its road path")
        rows.append(
            {
                "from_id": str(source),
                "from_type": str(source_data["type"]),
                "from_cluster": int(source_data["cluster"]),
                "to_id": str(target),
                "to_type": str(target_data["type"]),
                "to_cluster": int(target_data["cluster"]),
                # Keep established CSV aliases as well as explicit from/to names.
                "source_type": str(source_data["type"]),
                "source_cluster": int(source_data["cluster"]),
                "target_type": str(target_data["type"]),
                "target_cluster": int(target_data["cluster"]),
                "technical_coefficient": float(data["dag_weight"]),
                "road_distance": float(data["road_distance"]),
                "edge_weight": float(data["weight"]),
            }
        )
        geometries.append(road.path_geometry(road_path))

    edge_columns = [
        "from_id",
        "from_type",
        "from_cluster",
        "to_id",
        "to_type",
        "to_cluster",
        "source_type",
        "source_cluster",
        "target_type",
        "target_cluster",
        "technical_coefficient",
        "road_distance",
        "edge_weight",
    ]
    if rows:
        geo = gpd.GeoDataFrame(rows, geometry=geometries, crs=road.crs).to_crs(WGS84)
    else:
        geo = gpd.GeoDataFrame(
            {column: pd.Series(dtype="object") for column in edge_columns},
            geometry=gpd.GeoSeries([], crs=WGS84),
            crs=WGS84,
        )
    table_columns = [
        "source_type",
        "source_cluster",
        "target_type",
        "target_cluster",
        "technical_coefficient",
        "road_distance",
        "edge_weight",
    ]
    table = pd.DataFrame(geo.drop(columns="geometry"))[table_columns]
    return table, geo


def _cluster_layer(partitions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    clusters = partitions.copy()
    clusters["cluster_no"] = clusters["cluster"].astype(int)
    clusters["cluster_id"] = [
        make_node_id(str(type_value), int(cluster))
        for type_value, cluster in zip(clusters["type"], clusters["cluster"], strict=True)
    ]
    return clusters.to_crs(WGS84)


def write_spatial_geopackage(
    path: str | Path,
    partitions: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
) -> Path:
    """Write the three requested GIS layers into one portable GeoPackage."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    clusters = _cluster_layer(partitions)
    pyogrio.write_dataframe(
        clusters,
        target,
        layer="clusters",
        driver="GPKG",
        promote_to_multi=True,
    )
    pyogrio.write_dataframe(
        nodes,
        target,
        layer="nodes",
        driver="GPKG",
        append=True,
        geometry_type="Point",
    )
    pyogrio.write_dataframe(
        paths,
        target,
        layer="paths",
        driver="GPKG",
        append=True,
        geometry_type="LineString",
    )
    return target


def spatial_summary(
    *,
    partitions: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
) -> dict[str, object]:
    """Return compact run-specific numbers used by metadata and OUTPUTS.txt."""
    type_counts = partitions.groupby("type", sort=True).size() if len(partitions) else pd.Series()
    route_distance = (
        pd.to_numeric(paths["road_distance"], errors="coerce").to_numpy(float)
        if len(paths)
        else np.empty(0, dtype=float)
    )
    return {
        "represented_types": int(partitions["type"].astype(str).nunique()),
        "clusters_min_per_type": int(type_counts.min()) if len(type_counts) else 0,
        "clusters_median_per_type": float(type_counts.median()) if len(type_counts) else 0.0,
        "clusters_max_per_type": int(type_counts.max()) if len(type_counts) else 0,
        "x_in_degree_max": int(nodes["in_degree"].max()) if len(nodes) else 0,
        "x_out_degree_max": int(nodes["out_degree"].max()) if len(nodes) else 0,
        "x_in_degree_mean": float(nodes["in_degree"].mean()) if len(nodes) else 0.0,
        "x_out_degree_mean": float(nodes["out_degree"].mean()) if len(nodes) else 0.0,
        "route_distance_total": float(route_distance.sum()) if len(route_distance) else 0.0,
        "route_distance_min": float(route_distance.min()) if len(route_distance) else None,
        "route_distance_median": float(np.median(route_distance)) if len(route_distance) else None,
        "route_distance_max": float(route_distance.max()) if len(route_distance) else None,
        "spatial_output_crs": WGS84,
        "spatial_layers": ["clusters", "nodes", "paths"],
    }


def write_outputs_description(
    path: str | Path,
    *,
    metadata: Mapping[str, object],
    output_paths: Mapping[str, str],
) -> Path:
    """Write a stable human-readable description plus run-specific diagnostics."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def number(key: str, default: object = "n/a") -> object:
        value = metadata.get(key, default)
        if isinstance(value, float):
            return f"{value:,.6g}"
        if isinstance(value, int):
            return f"{value:,}"
        return value

    lines = [
        "SIGMA ENGINE OUTPUTS",
        "====================",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"SIGMA Engine version: {metadata.get('sigma_engine_version', 'unknown')}",
        f"Workflow: {metadata.get('workflow_version', 'unknown')}",
    ]
    if metadata.get("area_slug"):
        lines.append(
            "Area: "
            f"{metadata.get('area_name') or metadata.get('area_slug')} "
            f"({metadata.get('area_slug')})"
        )
    lines += [
        f"Classification: {metadata.get('classification', 'unknown')}",
        "",
        "PRIMARY GIS OUTPUT",
        "------------------",
        "sigma_spatial_outputs.gpkg is one GeoPackage containing all sectors:",
        "  clusters  Polygon/MultiPolygon cluster partitions for every type. Key fields:",
        "            type, cluster, cluster_no, cluster_id, plus Voronoi diagnostics.",
        "  nodes     Point representation of network X nodes in EPSG:4326. Key fields:",
        "            type, cluster_no, lat, lon, in_degree, out_degree, centrality.",
        "  paths     Exact shortest road-network route used for every connected X edge.",
        "            Key fields: from_type/from_cluster and to_type/to_cluster, road_distance,",
        "            technical_coefficient, and edge_weight. Geometry is the actual route,",
        "            not a straight or curved schematic chord.",
        "",
        "OTHER OUTPUTS",
        "-------------",
        "sigma_clustered_points.parquet: Step-2 classified points with "
        "network-HDBSCAN cluster labels.",
        "sigma_network_centers.parquet: exact network 1-median for each retained [type, cluster].",
        "sigma_points_with_center_distance.parquet: retained points with road distance "
        "to their median.",
        "sigma_partitions.parquet: native-CRS Step-4 cluster polygons; also represented "
        "in the GeoPackage.",
        "sigma_io_dag.csv: surviving sector-level MWAS DAG edges and weights.",
        "sigma_X_nodes.csv: tabular X nodes and centrality.",
        "sigma_X_edges.csv: tabular X edges and shortest-path distances.",
        "sigma_disconnected_overlap_pairs.csv: candidate overlaps whose centers are "
        "road-disconnected.",
        "sigma_points_centrality.parquet: final point-level SIGMA centrality scores.",
        "sigma_run_metadata.json: machine-readable run metadata and diagnostics.",
        "OUTPUTS.txt: this file.",
        "",
        "CURRENT RUN",
        "-----------",
        f"Scored points: {number('scored_points')}",
        f"Represented economic types: {number('represented_types')}",
        f"Clusters / X nodes: {number('x_nodes')}",
        f"X edges / shortest-road paths: {number('x_edges')}",
        f"Disconnected overlap pairs omitted from X: {number('disconnected_overlap_pairs')}",
        "Clusters per type (min / median / max): "
        f"{number('clusters_min_per_type')} / {number('clusters_median_per_type')} / "
        f"{number('clusters_max_per_type')}",
        "Maximum in-degree / out-degree: "
        f"{number('x_in_degree_max')} / {number('x_out_degree_max')}",
        "Mean in-degree / out-degree: "
        f"{number('x_in_degree_mean')} / {number('x_out_degree_mean')}",
        "Route distance total: "
        f"{number('route_distance_total')} {metadata.get('road_distance_unit', '')}",
        "Route distance min / median / max: "
        f"{number('route_distance_min')} / {number('route_distance_median')} / "
        f"{number('route_distance_max')} {metadata.get('road_distance_unit', '')}",
        f"Road-distance CRS: {metadata.get('road_distance_crs', 'unknown')}",
        f"GIS output CRS: {metadata.get('spatial_output_crs', WGS84)}",
        "",
        "FILES WRITTEN",
        "-------------",
    ]
    for name, value in sorted(output_paths.items()):
        lines.append(f"{name}: {value}")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
