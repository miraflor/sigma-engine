import math

import networkx as nx

from sigma_engine.spatial_graph import directed_eigenvector_centrality


def test_dag_centrality_uses_undirected_projection_and_is_finite():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c", weight=3.0)

    incoming = directed_eigenvector_centrality(graph, "incoming")
    outgoing = directed_eigenvector_centrality(graph, "outgoing")

    assert not incoming.degenerate_dag
    assert incoming.direction == "undirected"

    values = incoming.values
    assert set(values) == {"a", "b", "c"}
    assert all(math.isfinite(value) and value > 0.0 for value in values.values())

    # On the undirected weighted path a--2--b--3--c, b is the most central node.
    assert values["b"] > values["c"] > values["a"]

    # Directionality is deliberately removed before centrality.
    assert outgoing.direction == "undirected"
    for node in values:
        assert math.isclose(
            outgoing.values[node],
            incoming.values[node],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_zero_weight_edge_does_not_change_eigenvector_support():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=0.0)

    result = directed_eigenvector_centrality(graph, "incoming")

    expected = 1.0 / math.sqrt(2.0)
    assert math.isclose(result.values["a"], expected, rel_tol=1e-12)
    assert math.isclose(result.values["b"], expected, rel_tol=1e-12)
