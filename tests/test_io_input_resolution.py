from types import SimpleNamespace

import pandas as pd
import pytest

from sigma_engine.pipeline import _load_io_pair


def _z16():
    ids = [f"{i:02d}" for i in range(1, 17)]
    values = [[0.0] * 16 for _ in range(16)]
    values[0][1] = 20.0
    return pd.DataFrame(values, index=ids, columns=ids)


def test_bundled_z_and_total_output_derive_a_without_external_files(monkeypatch):
    z = _z16()
    x = pd.Series(100.0, index=z.columns, name="total_output")
    x.attrs["resource_filename"] = "psa_2018_io16_total_output.csv"
    x.attrs["resource_sha256"] = "abc123"

    monkeypatch.setattr(
        "sigma_engine.pipeline.load_engine_io_table",
        lambda *args, **kwargs: (z, SimpleNamespace(source_id="builtin")),
    )
    monkeypatch.setattr(
        "sigma_engine.pipeline.load_builtin_total_output",
        lambda *args, **kwargs: x,
    )

    transactions, coefficients, _ = _load_io_pair(
        classification="io16",
        transactions_override_path=None,
        transactions_sheet=0,
        technical_coefficients_path=None,
        technical_coefficients_sheet=0,
    )

    assert transactions is z
    assert coefficients.loc["01", "02"] == pytest.approx(0.2)
    assert (
        coefficients.attrs["technical_coefficient_source"]
        == "derived_from_builtin_psa_total_output"
    )
    assert (
        coefficients.attrs["technical_coefficient_source_path"]
        == "package:psa_2018_io16_total_output.csv"
    )
    assert coefficients.attrs["total_output_resource_sha256"] == "abc123"


def test_full_transaction_override_derives_a_when_explicit_a_is_absent(
    monkeypatch, tmp_path
):
    z = _z16()
    transaction_path = tmp_path / "full.xlsx"
    transaction_path.touch()
    x = pd.Series(100.0, index=z.columns)

    monkeypatch.setattr(
        "sigma_engine.pipeline.load_engine_io_table",
        lambda *args, **kwargs: (z, SimpleNamespace(source_id="override")),
    )
    monkeypatch.setattr(
        "sigma_engine.pipeline.read_total_output_vector",
        lambda *args, **kwargs: x,
    )

    transactions, coefficients, _ = _load_io_pair(
        classification="io16",
        transactions_override_path=str(transaction_path),
        transactions_sheet=0,
        technical_coefficients_path=None,
        technical_coefficients_sheet=0,
    )

    assert transactions is z
    assert coefficients.loc["01", "02"] == pytest.approx(0.2)
    assert (
        coefficients.attrs["technical_coefficient_source"]
        == "derived_from_transaction_total_output"
    )
    assert coefficients.attrs["technical_coefficient_source_path"] == str(
        transaction_path.resolve()
    )
