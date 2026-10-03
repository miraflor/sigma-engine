from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, box

from sigma_engine.network import AugmentedNetwork
from sigma_engine.outputs import (
    build_x_edge_outputs,
    build_x_node_outputs,
    write_spatial_geopackage,
)


def _road() -> AugmentedNetwork:
    nodes = gpd.GeoDataFrame(
        {"network_node": [0, 1, 2]},
        geometry=[Point(0, 0), Point(100, 0), Point(200, 0)],
        crs="EPSG:3857",
    )
    edges = gpd.GeoDataFrame(
        {
            "u": [0, 1],
            "v": [1, 2],
            "length": [100.0, 100.0],
            "parent_edge_id": [0, 1],
            "aug_edge_id": [0, 1],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
        ],
        crs="EPSG:3857",
    )
    return AugmentedNetwork.from_checkpoint_frames(
        nodes, edges, np.asarray([], dtype=np.int64)
    )


def test_shortest_path_geometry_preserves_road_route():
    road = _road()
    distances, paths = road.shortest_paths_to_targets(0, [2])
    assert distances.tolist() == [200.0]
    assert paths == [(0, 1, 2)]
    geometry = road.path_geometry(paths[0])
    assert list(geometry.coords) == [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]


def test_single_geopackage_contains_clusters_nodes_and_paths(tmp_path: Path):
    road = _road()
    _, paths = road.shortest_paths_to_targets(0, [2])

    graph = nx.DiGraph()
    graph.add_node("a", type="A", cluster=0, center_x=0.0, center_y=0.0)
    graph.add_node("b", type="B", cluster=1, center_x=200.0, center_y=0.0)
    graph.add_edge(
        "a",
        "b",
        dag_weight=0.4,
        road_distance=200.0,
        weight=80.0,
        road_path=paths[0],
    )
    _, node_geo = build_x_node_outputs(graph, {"a": 0.25, "b": 0.75}, road.crs)
    _, path_geo = build_x_edge_outputs(graph, road)
    partitions = gpd.GeoDataFrame(
        {"type": ["A", "B"], "cluster": [0, 1]},
        geometry=[box(0, -10, 100, 10), box(100, -10, 200, 10)],
        crs=road.crs,
    )

    target = write_spatial_geopackage(
        tmp_path / "sigma_spatial_outputs.gpkg", partitions, node_geo, path_geo
    )
    layers = set(gpd.list_layers(target)["name"])
    assert layers == {"clusters", "nodes", "paths"}

    nodes = gpd.read_file(target, layer="nodes")
    assert {"type", "cluster_no", "lat", "lon", "in_degree", "out_degree"} <= set(
        nodes.columns
    )
    assert nodes.loc[nodes["type"] == "A", "out_degree"].iloc[0] == 1
    assert nodes.loc[nodes["type"] == "B", "in_degree"].iloc[0] == 1

    routes = gpd.read_file(target, layer="paths")
    assert routes.loc[0, "from_type"] == "A"
    assert routes.loc[0, "to_type"] == "B"
    assert routes.loc[0, "road_distance"] == 200.0
