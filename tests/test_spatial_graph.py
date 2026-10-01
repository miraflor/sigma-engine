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
