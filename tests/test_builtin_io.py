import math

from sigma_engine.builtin_io import _canonical_text_sha256, load_engine_io_table


def test_builtin_io80_is_complete_canonical_psa_matrix():
    table, info = load_engine_io_table("io80")

    assert table.shape == (80, 80)
    assert list(table.index) == [f"{i:02d}" for i in range(1, 81)]
    assert list(table.columns) == list(table.index)
    assert table.attrs["source_layout"] == "builtin-canonical"
    assert info.source_id == "psa-2018-io80-transaction"
    assert info.source_kind == "builtin"
    assert info.reference_year == 2018
    assert info.source_path is None

    # Sentinels guard against accidentally bundling a reordered or rounded matrix.
    assert math.isclose(table.loc["01", "01"], 20129.315047509437, rel_tol=1e-12)
    assert math.isclose(table.loc["22", "36"], 234658.5424071815, rel_tol=1e-12)
    assert math.isclose(table.loc["80", "75"], 2452.599491776884, rel_tol=1e-12)
    assert table.loc["71", "71"] == 0.0


def test_builtin_io16_matches_same_2018_intermediate_use_total():
    io16, info16 = load_engine_io_table("io16")
    io80, _ = load_engine_io_table("io80")

    assert io16.shape == (16, 16)
    assert list(io16.index) == [f"{i:02d}" for i in range(1, 17)]
    assert info16.source_id == "psa-2018-io16-transaction"
    assert math.isclose(io16.loc["01", "03"], 1335493.450539227, rel_tol=1e-12)
    assert math.isclose(
        float(io16.to_numpy().sum()),
        float(io80.to_numpy().sum()),
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def test_custom_io_override_is_explicit(tmp_path):
    path = tmp_path / "custom.csv"
    sectors = [f"{i:02d}" for i in range(1, 17)]
    import pandas as pd

    frame = pd.DataFrame(0.0, index=sectors, columns=sectors)
    frame.loc["01", "02"] = 7.0
    frame.to_csv(path)

    table, info = load_engine_io_table("io16", override_path=path)
    assert table.loc["01", "02"] == 7.0
    assert info.source_kind == "override"
    assert info.source_id == "custom-override"
    assert info.source_path == str(path.resolve())


def test_builtin_resource_integrity_is_newline_independent():
    lf = b"sector,01,02\n01,1,2\n02,3,4\n"
    crlf = lf.replace(b"\n", b"\r\n")

    assert _canonical_text_sha256(lf) == _canonical_text_sha256(crlf)


def test_builtin_io80_fast_mwas_is_acyclic_and_accounts_for_removed_weight():
    import networkx as nx

    from sigma_engine.io_dag import fast_mwas, io_network

    table, _ = load_engine_io_table("io80")
    source = io_network(table)
    result = fast_mwas(source)

    assert nx.is_directed_acyclic_graph(result.graph)
    assert result.method == "fast_greedy"
    assert not result.optimal
    for u, v, data in result.graph.edges(data=True):
        assert source.has_edge(u, v)
        assert float(data["weight"]) == float(source[u][v]["weight"])
    source_weight = sum(float(d["weight"]) for _, _, d in source.edges(data=True))
    assert math.isclose(
        result.retained_weight + result.removed_weight,
        source_weight,
        rel_tol=0.0,
        abs_tol=1e-8,
    )
