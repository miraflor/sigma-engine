import math

import networkx as nx

from sigma_engine.spatial_graph import directed_eigenvector_centrality


def test_dag_centrality_is_explicitly_degenerate_and_finite():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c", weight=3.0)

    result = directed_eigenvector_centrality(graph, "incoming")

    assert result.degenerate_dag
    assert result.values == {"a": 0.0, "b": 0.0, "c": 1.0}


def test_zero_weight_edge_does_not_change_dag_eigenvector_support():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=0.0)

    result = directed_eigenvector_centrality(graph, "incoming")

    expected = 1.0 / math.sqrt(2.0)
    assert result.values == {"a": expected, "b": expected}
