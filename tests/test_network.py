import geopandas as gpd
from shapely.geometry import LineString, Point

from sigma_engine.network import RoadNetwork


def test_continuous_snap_and_distance():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])], crs="EPSG:3857"
    )
    points = gpd.GeoDataFrame(geometry=[Point(2, 1), Point(18, -1)], crs=roads.crs)
    base = RoadNetwork.from_geodataframe(roads)
    snapped = base.snap(points)
    aug = base.augment(snapped)
    a, b = aug.point_node
    assert abs(aug.distance(int(a), int(b)) - 16.0) < 1e-9
    assert snapped.frame.snap_distance.tolist() == [1.0, 1.0]


def test_duplicate_reversed_road_arcs_are_collapsed_and_snap_ties_are_stable():
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (0, 0)])],
        crs="EPSG:3857",
    )
    network = RoadNetwork.from_geodataframe(roads)
    assert len(network.segments) == 1
    points = gpd.GeoDataFrame(geometry=[Point(5, 1)], crs=roads.crs)
    snapped = network.snap(points)
    assert int(snapped.frame.iloc[0].edge_id) == 0


def test_reversing_source_line_does_not_change_augmented_nodes_or_median_geometry():
    from sigma_engine.center import network_1_median

    points = gpd.GeoDataFrame(
        geometry=[Point(2, 0), Point(8, 0)],
        crs="EPSG:3857",
    )
    forward = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=points.crs,
    )
    reverse = gpd.GeoDataFrame(
        geometry=[LineString([(10, 0), (0, 0)])],
        crs=points.crs,
    )

    net1 = RoadNetwork.from_geodataframe(forward)
    aug1 = net1.augment(net1.snap(points))
    net2 = RoadNetwork.from_geodataframe(reverse)
    aug2 = net2.augment(net2.snap(points))

    assert aug1.nodes.geometry.tolist() == aug2.nodes.geometry.tolist()
    assert network_1_median(aug1, aug1.point_node).geometry.equals(
        network_1_median(aug2, aug2.point_node).geometry
    )
