import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from sigma_engine.scoring import tempered_point_scores
from sigma_engine.spatial_graph import make_node_id


def test_tempered_score_uses_cluster_p90_and_is_bounded():
    points = gpd.GeoDataFrame(
        {
            "canonical_id": [f"p{i}" for i in range(5)],
            "type": ["A"] * 5,
            "cluster": [0] * 5,
            "distance_to_cluster_median": [0.0, 10.0, 20.0, 30.0, 100.0],
        },
        geometry=[Point(i, 0) for i in range(5)],
        crs="EPSG:3857",
    )
    centrality = {make_node_id("A", 0): 2.0}
    out = tempered_point_scores(points, centrality, distance_tempering=0.15)

    expected_p90 = float(np.quantile(np.array([0, 10, 20, 30, 100], dtype=float), 0.90))
    assert np.allclose(out["cluster_distance_p90"], expected_p90)
    assert out.loc[0, "sigma_score"] == 2.0
    assert out["sigma_score"].min() >= 1.7 - 1e-12
    assert out["sigma_score"].max() <= 2.0 + 1e-12
    assert out.loc[4, "normalized_cluster_distance"] == 1.0
    assert out.loc[4, "sigma_score"] == 1.7


def test_zero_tempering_makes_all_points_equal_to_cluster_centrality():
    points = gpd.GeoDataFrame(
        {"type": ["A", "A"], "cluster": [0, 0], "distance_to_cluster_median": [0.0, 999.0]},
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:3857",
    )
    out = tempered_point_scores(
        points,
        {make_node_id("A", 0): 3.0},
        distance_tempering=0.0,
    )
    assert out["sigma_score"].tolist() == [3.0, 3.0]
