import geopandas as gpd
import pytest
from shapely.geometry import Point

from sigma_engine.pipeline import _clean_center_output, _validate_common


def test_revised_defaults_validate():
    _validate_common(
        classification="io80",
        mwas_method="fast",
        centrality_direction="incoming",
        distance_tempering=0.15,
    )


def test_distance_tempering_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="distance_tempering"):
        _validate_common(
            classification="io80",
            mwas_method="fast",
            centrality_direction="incoming",
            distance_tempering=1.01,
        )


def test_centers_remain_one_row_per_cluster():
    centers = gpd.GeoDataFrame(
        {
            "type": ["A"],
            "cluster": [3],
            "center_network_node": [8],
            "median_objective": [100.0],
        },
        geometry=[Point(1, 2)],
        crs="EPSG:3857",
    )
    out = _clean_center_output(centers)
    assert list(out.columns) == ["type", "cluster", "center_network_node", "geometry"]
    assert len(out) == 1
