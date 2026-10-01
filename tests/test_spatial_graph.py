import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, box

from sigma_engine.network import RoadNetwork
from sigma_engine.spatial_graph import instantiate_network, make_node_id


def test_spatial_instantiation_preserves_io_weight_distance_and_product():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (20, 0)])],
        crs="EPSG:3857",
    )
    point_frame = gpd.GeoDataFrame(
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    base = RoadNetwork.from_geodataframe(roads)
    augmented = base.augment(base.snap(point_frame))

    centers = gpd.GeoDataFrame(
        {
            "type": ["A", "B"],
            "cluster": [0, 0],
            "center_network_node": augmented.point_node,
        },
        geometry=[Point(2, 0), Point(18, 0)],
        crs=roads.crs,
    )
    # Positive-area overlap is deliberate: both type partitions cover the same surface.
    partitions = gpd.GeoDataFrame(
        {
            "type": ["A", "B"],
            "cluster": [0, 0],
        },
        geometry=[box(0, -2, 20, 2), box(0, -2, 20, 2)],
        crs=roads.crs,
    )
    io_dag = nx.DiGraph()
    io_dag.add_edge("A", "B", weight=3.0)

    graph, diagnostics = instantiate_network(centers, partitions, io_dag, augmented)

    source = make_node_id("A", 0)
    target = make_node_id("B", 0)
    edge = graph[source][target]
    assert edge["dag_weight"] == 3.0
    assert edge["road_distance"] == 16.0
    assert edge["weight"] == 48.0
    assert diagnostics.empty


def test_spatial_instantiation_uses_targeted_distances_and_reuses_shared_source_node():
    centers = gpd.GeoDataFrame(
        {
            "type": ["A", "A", "B", "C"],
            "cluster": [0, 1, 0, 0],
            "center_network_node": [10, 10, 20, 30],
        },
        geometry=[Point(1, 0), Point(9, 0), Point(5, 0), Point(5, 0)],
        crs="EPSG:3857",
    )
    partitions = gpd.GeoDataFrame(
        {
            "type": ["A", "A", "B", "C"],
            "cluster": [0, 1, 0, 0],
        },
        geometry=[
            box(0, 0, 5, 10),
            box(5, 0, 10, 10),
            box(0, 0, 10, 10),
            box(0, 0, 10, 10),
        ],
        crs="EPSG:3857",
    )
    io_dag = nx.DiGraph()
    io_dag.add_edge("A", "B", weight=2.0)
    io_dag.add_edge("A", "C", weight=3.0)

    class TargetOnlyRoad:
        def __init__(self):
            self.calls = []

        def distances_to_targets(self, source, targets):
            targets = list(map(int, targets))
            self.calls.append((int(source), tuple(targets)))
            return __import__("numpy").abs(__import__("numpy").asarray(targets, dtype=float) - float(source))

        def distances_from(self, source):  # pragma: no cover - must never be called
            raise AssertionError("network X must not materialize full distance dictionaries")

    road = TargetOnlyRoad()
    graph, diagnostics = instantiate_network(centers, partitions, io_dag, road)

    # Both A clusters share road node 10, so one targeted shortest-path evaluation serves
    # all four spatial overlap edges across both outgoing IO relationships.
    assert len(road.calls) == 1
    assert road.calls[0][0] == 10
    assert sorted(road.calls[0][1]) == [20, 20, 30, 30]
    assert graph.number_of_edges() == 4
    assert diagnostics.empty


def test_network_x_checkpoint_restore_round_trips_tabular_artifacts(tmp_path):
    import pandas as pd

    from sigma_engine.pipeline import _restore_network_x

    nodes = tmp_path / "nodes.csv"
    edges = tmp_path / "edges.csv"
    disconnected = tmp_path / "disconnected.csv"
    pd.DataFrame(
        [
            {
                "node_id": make_node_id("01", 0),
                "type": "01",
                "cluster": 0,
                "center_network_node": 4,
                "center_x": 1.0,
                "center_y": 2.0,
                "eigenvector_centrality": None,
            },
            {
                "node_id": make_node_id("02", 0),
                "type": "02",
                "cluster": 0,
                "center_network_node": 9,
                "center_x": 3.0,
                "center_y": 4.0,
                "eigenvector_centrality": None,
            },
        ]
    ).to_csv(nodes, index=False)
    pd.DataFrame(
        [
            {
                "node_from": make_node_id("01", 0),
                "node_to": make_node_id("02", 0),
                "dag_weight": 2.0,
                "road_distance": 5.0,
                "weight": 10.0,
            }
        ]
    ).to_csv(edges, index=False)
    pd.DataFrame(
        columns=["type1", "cluster1", "type2", "cluster2", "status"]
    ).to_csv(disconnected, index=False)

    graph, diagnostics = _restore_network_x(nodes, edges, disconnected)

    assert graph.nodes[make_node_id("01", 0)]["type"] == "01"
    assert graph[make_node_id("01", 0)][make_node_id("02", 0)]["weight"] == 10.0
    assert diagnostics.empty
