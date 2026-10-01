import geopandas as gpd
import pytest
from shapely.geometry import Point

from sigma_engine.point_input import prepare_point_input


def _frame(**columns):
    base = {
        "canonical_id": ["p1", "p2"],
        "canonical_name": ["One", "Two"],
    }
    base.update(columns)
    return gpd.GeoDataFrame(
        base,
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:4326",
    )


def test_auto_detects_io_code_convention():
    points, info = prepare_point_input(
        _frame(io80_code=[1, "02"]),
        classification="io80",
    )
    assert info.classification_column == "io80_code"
    assert points["type"].tolist() == ["01", "02"]


def test_auto_detects_map_safe_placetype_convention():
    points, info = prepare_point_input(
        _frame(io80_map_code=["01", 2]),
        classification="io80",
    )
    assert info.classification_column == "io80_map_code"
    assert points["type"].tolist() == ["01", "02"]


def test_plural_candidate_sets_are_not_auto_selected():
    with pytest.raises(ValueError, match="could not auto-detect"):
        prepare_point_input(
            _frame(io80_codes=["01|02", "02"]),
            classification="io80",
        )


def test_recognized_columns_must_not_disagree():
    with pytest.raises(ValueError, match="disagree"):
        prepare_point_input(
            _frame(io80_code=["01", "02"], io80_map_code=["01", "03"]),
            classification="io80",
        )


def test_consistent_columns_choose_the_more_complete_field():
    points, info = prepare_point_input(
        _frame(io80_code=["01", None], io80_map_code=["01", "02"]),
        classification="io80",
    )
    assert info.classification_column == "io80_map_code"
    assert points["type"].tolist() == ["01", "02"]


def test_custom_single_value_column_can_be_selected_explicitly():
    points, info = prepare_point_input(
        _frame(my_sector=["01", "02"]),
        classification="io80",
        io80_column="my_sector",
    )
    assert info.classification_column == "my_sector"
    assert points["type"].tolist() == ["01", "02"]


def test_compact_gis_export_without_name_uses_id_as_display_fallback():
    frame = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1", "p2"],
            "io16_map_code": ["01", "02"],
        },
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:4326",
    )
    points, info = prepare_point_input(frame, classification="io16")
    assert info.canonical_name_source == "canonical_id"
    assert points["canonical_name"].tolist() == ["p1", "p2"]


def test_point_input_rejects_non_point_geometry():
    frame = gpd.GeoDataFrame(
        {
            "canonical_id": ["p1"],
            "canonical_name": ["One"],
            "io80_code": ["01"],
        },
        geometry=gpd.GeoSeries.from_wkt(["LINESTRING (0 0, 1 0)"]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="only Point geometry"):
        prepare_point_input(frame, classification="io80")
