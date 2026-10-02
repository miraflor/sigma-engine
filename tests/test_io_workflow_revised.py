import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sigma_engine.io_workflow import (
    derive_technical_coefficients,
    read_total_output_vector,
    reweight_surviving_graph,
)


def _matrix(values):
    return pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"], dtype=float)


def test_surviving_edges_keep_z_for_audit_but_use_a_as_weight():
    z = _matrix([[0, 100, 1], [2, 0, 50], [40, 3, 0]])
    a = _matrix([[0, 0.20, 0.01], [0.02, 0, 0.30], [0.40, 0.03, 0]])
    surviving = nx.DiGraph()
    surviving.add_nodes_from(["A", "B", "C"])
    surviving.add_edge("A", "B", weight=100.0)
    surviving.add_edge("B", "C", weight=50.0)

    dag, rows = reweight_surviving_graph(surviving, z, a, mwas_method="fast")

    assert nx.is_directed_acyclic_graph(dag)
    assert dag["A"]["B"]["transaction_value"] == 100.0
    assert dag["A"]["B"]["technical_coefficient"] == 0.20
    assert dag["A"]["B"]["weight"] == 0.20
    assert list(rows.columns) == [
        "source_type",
        "target_type",
        "transaction_value",
        "technical_coefficient",
        "mwas_method",
    ]


def test_matrix_sector_mismatch_is_not_silently_dropped():
    z = _matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    a = pd.DataFrame([[0, 0.2], [0.1, 0]], index=["A", "B"], columns=["A", "B"])
    surviving = nx.DiGraph([("A", "B")])
    with pytest.raises(ValueError, match="exactly the same sectors"):
        reweight_surviving_graph(surviving, z, a, mwas_method="fast")


def test_derive_a_uses_gross_output_not_intermediate_column_sum():
    z = _matrix([[10, 20, 30], [5, 10, 15], [1, 2, 3]])
    x = pd.Series({"A": 100.0, "B": 200.0, "C": 300.0})

    a = derive_technical_coefficients(z, x)

    assert a.loc["A", "A"] == pytest.approx(0.10)
    assert a.loc["A", "B"] == pytest.approx(0.10)
    assert a.loc["A", "C"] == pytest.approx(0.10)
    assert not np.allclose(a.to_numpy(), z.div(z.sum(axis=0), axis="columns").to_numpy())
    assert a.attrs["technical_coefficient_source"] == "derived_from_transaction_total_output"


def _write_presentation_workbook(path, *, include_total_output=True):
    n = 16
    codes = [f"{i:02d}" for i in range(1, n + 1)]
    z = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
    x = z.sum(axis=0) + np.arange(1_000.0, 1_000.0 + n)

    columns = 2 + n + 4
    raw = pd.DataFrame(np.nan, index=range(n + 4), columns=range(columns), dtype=object)
    raw.iloc[0, 2 : 2 + n] = codes
    raw.iloc[1, 1] = "Description"
    raw.iloc[1, 2 : 2 + n] = [f"Sector {code}" for code in codes]
    total_output_column = 2 + n + 3
    if include_total_output:
        raw.iloc[1, total_output_column] = "Total Output"

    for row, code in enumerate(codes, start=2):
        raw.iloc[row, 0] = code
        raw.iloc[row, 1] = f"Sector {code}"
        raw.iloc[row, 2 : 2 + n] = z[row - 2]
        if include_total_output:
            raw.iloc[row, total_output_column] = x[row - 2]

    raw.to_excel(path, index=False, header=False)
    return codes, z, x


def test_total_output_is_read_from_same_presentation_rows(tmp_path):
    workbook = tmp_path / "full_io.xlsx"
    codes, z, x = _write_presentation_workbook(workbook)

    output = read_total_output_vector(workbook, sector_ids=codes)
    transactions = pd.DataFrame(z, index=codes, columns=codes)
    a = derive_technical_coefficients(transactions, output)

    np.testing.assert_allclose(output.to_numpy(), x)
    np.testing.assert_allclose(a.to_numpy(), z / x[None, :])


def test_missing_total_output_is_rejected_instead_of_using_column_sums(tmp_path):
    workbook = tmp_path / "intermediate_only.xlsx"
    codes, _, _ = _write_presentation_workbook(workbook, include_total_output=False)

    with pytest.raises(ValueError, match="Total Output"):
        read_total_output_vector(workbook, sector_ids=codes)
