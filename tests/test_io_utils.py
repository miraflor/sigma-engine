from sigma_engine.io_utils import normalize_sector_id


def test_sector_id_normalization_matches_sigma_codes():
    assert normalize_sector_id("01") == "01"
    assert normalize_sector_id(1) == "01"
    assert normalize_sector_id(1.0) == "01"
    assert normalize_sector_id("1.0") == "01"
    assert normalize_sector_id("80") == "80"
    assert normalize_sector_id("A") == "A"
    assert normalize_sector_id("  ") is None


def test_sector_id_normalization_does_not_round_through_float():
    assert normalize_sector_id("12345678901234567890.0") == "12345678901234567890"
