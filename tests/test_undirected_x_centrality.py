import math

import networkx as nx

from sigma_engine.spatial_graph import directed_eigenvector_centrality


def test_direction_is_removed_before_eigenvector_centrality():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=1.0)

    incoming = directed_eigenvector_centrality(graph, "incoming")
    outgoing = directed_eigenvector_centrality(graph, "outgoing")

    assert incoming.direction == "undirected"
    assert outgoing.direction == "undirected"
    assert not incoming.degenerate_dag
    assert incoming.values == outgoing.values
    assert incoming.values["B"] > incoming.values["A"]
    assert math.isclose(incoming.values["A"], incoming.values["C"], rel_tol=1e-9)


def test_reciprocal_directed_weights_are_combined_in_projection():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=2.0)
    graph.add_edge("B", "A", weight=3.0)

    result = directed_eigenvector_centrality(graph)

    assert result.direction == "undirected"
    assert math.isclose(result.values["A"], result.values["B"], rel_tol=1e-9)


def test_zero_weight_edges_do_not_drive_centrality():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=0.0)
    graph.add_edge("B", "C", weight=0.0)

    result = directed_eigenvector_centrality(graph)

    expected = 1.0 / math.sqrt(3.0)
    assert all(math.isclose(value, expected, rel_tol=1e-12) for value in result.values.values())
