import networkx as nx
import pandas as pd
import pytest

from sigma_engine.io_workflow import reweight_surviving_graph


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
