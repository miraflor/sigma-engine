import networkx as nx
import numpy as np
import pandas as pd
import pytest

from sigma_engine.io_dag import (
    exact_mwas,
    fast_mwas,
    io_network,
    maximum_weight_acyclic_subgraph,
    read_io_table,
    solve_mwas,
)


def _weight(graph: nx.DiGraph) -> float:
    return float(sum(float(data["weight"]) for _, _, data in graph.edges(data=True)))


def test_fast_mwas_is_deterministic_acyclic_and_prefers_heavy_edges():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(
        [("A", "B", 10.0), ("B", "C", 9.0), ("C", "A", 1.0), ("C", "B", 2.0)]
    )
    a = fast_mwas(graph)
    b = fast_mwas(graph)
    assert a.method == "fast_greedy"
    assert not a.exact and not a.optimal
    assert nx.is_directed_acyclic_graph(a.graph)
    assert sorted(a.graph.edges(data="weight")) == sorted(b.graph.edges(data="weight"))
    assert a.graph.has_edge("A", "B")
    assert a.graph.has_edge("B", "C")
    assert not a.graph.has_edge("C", "A")
    assert a.retained_weight + a.removed_weight == pytest.approx(_weight(graph))


def test_exact_mwas_finds_optimum_on_three_cycle():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("A", "B", 20.0), ("B", "C", 30.0), ("C", "A", 10.0)])
    result = exact_mwas(graph)
    assert result.method == "exact_milp"
    assert result.exact and result.optimal
    assert nx.is_directed_acyclic_graph(result.graph)
    assert result.retained_weight == pytest.approx(50.0)
    assert result.removed_weight == pytest.approx(10.0)


def test_fast_is_default_and_exact_is_explicit():
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("A", "B", 2.0), ("B", "A", 1.0)])
    assert solve_mwas(graph).method == "fast_greedy"
    assert maximum_weight_acyclic_subgraph(graph).has_edge("A", "B")
    assert solve_mwas(graph, method="exact").method == "exact_milp"
    with pytest.raises(ValueError, match="fast.*exact"):
        solve_mwas(graph, method="other")


def test_mwas_self_loops_are_counted_as_removed():
    graph = nx.DiGraph()
    graph.add_edge("A", "A", weight=5.0)
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "B", weight=4.0)
    result = fast_mwas(graph)
    assert not any(u == v for u, v in result.graph.edges)
    assert result.self_loop_weight == pytest.approx(9.0)
    assert result.retained_weight == pytest.approx(1.0)
    assert result.removed_weight == pytest.approx(9.0)


def test_mwas_rejects_invalid_weights():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=float("nan"))
    with pytest.raises(ValueError, match="invalid"):
        fast_mwas(graph)


def test_io_numeric_codes_are_normalized_to_two_digits(tmp_path):
    path = tmp_path / "io.csv"
    pd.DataFrame([[0, 2], [1, 0]], index=[1, 2], columns=[1, 2]).to_csv(path)

    table = read_io_table(path)

    assert list(table.index) == ["01", "02"]
    assert list(table.columns) == ["01", "02"]


def test_io_table_rejects_mismatched_sector_axes(tmp_path):
    path = tmp_path / "io.csv"
    pd.DataFrame([[0, 2], [1, 0]], index=["A", "B"], columns=["A", "C"]).to_csv(path)

    with pytest.raises(ValueError, match="identical normalized row and column"):
        read_io_table(path)


def test_io_table_rejects_nonfinite_weights(tmp_path):
    path = tmp_path / "io.csv"
    pd.DataFrame([[0.0, np.inf], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"]).to_csv(path)

    with pytest.raises(ValueError, match="non-finite"):
        read_io_table(path)


def test_io_presentation_workbook_extracts_expected_block(tmp_path):
    path = tmp_path / "io16.xlsx"
    labels = [f"Published sector {i}" for i in range(1, 17)]
    raw = pd.DataFrame([[None] * 22 for _ in range(24)])
    raw.iat[0, 0] = "2018 Input-Output Accounts"
    # Published labels appear horizontally above the matrix and vertically at left.
    for i, label in enumerate(labels):
        raw.iat[4, 2 + i] = label
        raw.iat[6 + i, 0] = label
        for j in range(16):
            raw.iat[6 + i, 2 + j] = float((i + 1) * 100 + (j + 1))
    # Extra publication columns must not be mistaken for intermediate demand.
    raw.iloc[6:22, 18:22] = 999999.0
    raw.to_excel(path, index=False, header=False)

    table = read_io_table(path, expected_sector_count=16)

    assert table.shape == (16, 16)
    assert list(table.index) == [f"{i:02d}" for i in range(1, 17)]
    assert list(table.columns) == list(table.index)
    assert table.loc["01", "01"] == 101.0
    assert table.loc["16", "16"] == 1616.0
    assert table.attrs["source_layout"] == "presentation-published-order"

