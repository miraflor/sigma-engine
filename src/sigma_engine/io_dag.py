"""Input-output loading and maximum-weight acyclic subgraph (MWAS) extraction.

SIGMA needs an acyclic sector graph before spatial instantiation.  The weighted
maximum acyclic subgraph problem keeps a subset of directed IO edges with maximum
total retained weight subject to acyclicity; equivalently, its complement is a
minimum-weight feedback arc set.  The weighted problem is NP-hard in general.

The production default is therefore a deterministic *fast heuristic*: process IO
edges in descending transaction weight and keep an edge iff it does not create a
cycle.  This is a SIGMA engineering heuristic; no approximation guarantee is
claimed for this particular greedy ordering.  The explicit exact mode instead
uses a mixed-integer topological-rank formulation solved by SciPy/HiGHS with zero
relative MIP gap.

Algorithmic background:
- Hassin & Rubinstein (1994), "Approximations for the maximum acyclic subgraph
  problem", Information Processing Letters 51(3), 133--140.
- Charon & Hudry (2006), "A branch-and-bound algorithm to solve the linear
  ordering problem for weighted tournaments", Discrete Applied Mathematics
  154(15), 2097--2116, discusses the equivalence among maximum acyclic
  subdigraph / minimum feedback arc set / linear ordering formulations.

The exact MILP below is the standard topological-order construction: each retained
edge ``u -> v`` forces ``rank(u) + 1 <= rank(v)``.  The specific descending-weight
greedy rule is intentionally documented as package-specific rather than falsely
attributed to those papers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .io_utils import normalize_sector_id


@dataclass(frozen=True)
class MWASResult:
    """Acyclic IO subgraph plus optimization/audit metadata.

    ``method="fast_greedy"`` is deterministic and deliberately heuristic: it
    scans edges from largest to smallest transaction weight and accepts an edge
    exactly when doing so does not create a directed cycle.  ``method="exact_milp"``
    solves the weighted maximum-acyclic-subgraph objective to proven optimality
    with a mixed-integer topological-order formulation.
    """

    graph: nx.DiGraph
    method: str
    exact: bool
    optimal: bool
    solver_status: str | None
    source_weight: float
    retained_weight: float
    removed_weight: float
    self_loop_weight: float
    source_edge_count: int
    retained_edge_count: int
    removed_edge_count: int


def _normalized_axis(values, *, axis_name: str) -> list[str]:
    """Normalize one IO-table axis and reject blank/duplicate sector IDs."""
    normalized = [normalize_sector_id(value) for value in values]
    if any(value is None for value in normalized):
        raise ValueError(f"IO {axis_name} sector identifiers must be non-blank")
    out = [str(value) for value in normalized]
    counts = Counter(out)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"IO {axis_name} sector identifiers are not unique after normalization: "
            f"{duplicates[:10]}"
        )
    return out


def _validated_transaction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate one already-extracted square transaction block."""
    if frame.empty:
        raise ValueError("IO transaction table is empty")

    row_ids = _normalized_axis(frame.index, axis_name="row")
    column_ids = _normalized_axis(frame.columns, axis_name="column")
    frame = frame.copy()
    frame.index = row_ids
    frame.columns = column_ids

    row_set = set(row_ids)
    column_set = set(column_ids)
    if row_set != column_set or len(row_ids) != len(column_ids):
        only_rows = sorted(row_set - column_set)
        only_columns = sorted(column_set - row_set)
        raise ValueError(
            "IO transaction block must be square with identical normalized row and column "
            f"sector IDs; only in rows={only_rows[:10]}, only in columns={only_columns[:10]}"
        )

    frame = frame.loc[row_ids, row_ids]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("IO transaction block contains missing or non-finite values")
    if (values < 0).any():
        raise ValueError("IO transaction weights must be non-negative")
    return frame.astype(float)


def _display_token(value: object) -> str | None:
    """Comparable token for labels in presentation-style spreadsheets."""
    if value is None or pd.isna(value):
        return None
    token = " ".join(str(value).split()).strip().casefold()
    return token or None


def _extract_presentation_block(raw: pd.DataFrame, sector_count: int) -> pd.DataFrame:
    """Locate an ``N x N`` intermediate-transaction block in a formatted worksheet.

    PSA-style workbooks place titles, units, descriptions, totals and final-demand columns
    around the transaction matrix.  The robust structural signal is that the same ordered
    sector labels appear once horizontally above the numeric block and once vertically to
    its left.  They may be industry codes *or* descriptions; SIGMA needs only their order.
    """
    if sector_count < 2:
        raise ValueError("presentation IO sector_count must be >= 2")
    rows, columns = raw.shape
    if rows < sector_count or columns < sector_count:
        raise ValueError(
            f"worksheet is too small to contain a {sector_count}x{sector_count} transaction block"
        )

    candidates: list[tuple[int, int, int, int, pd.DataFrame, list[object]]] = []
    max_header_row = rows - sector_count
    max_data_column = columns - sector_count

    for header_row in range(max_header_row + 1):
        for data_column in range(max_data_column + 1):
            horizontal_raw = raw.iloc[
                header_row, data_column : data_column + sector_count
            ].tolist()
            horizontal = [_display_token(value) for value in horizontal_raw]
            if any(token is None for token in horizontal) or len(set(horizontal)) != sector_count:
                continue

            first = horizontal[0]
            # Row labels for a transaction block must be to the left of its numeric cells.
            for label_column in range(data_column):
                possible_starts = [
                    row
                    for row in range(header_row + 1, rows - sector_count + 1)
                    if _display_token(raw.iat[row, label_column]) == first
                ]
                for data_row in possible_starts:
                    vertical_raw = raw.iloc[
                        data_row : data_row + sector_count, label_column
                    ].tolist()
                    vertical = [_display_token(value) for value in vertical_raw]
                    if vertical != horizontal:
                        continue

                    block = raw.iloc[
                        data_row : data_row + sector_count,
                        data_column : data_column + sector_count,
                    ].apply(pd.to_numeric, errors="coerce")
                    values = block.to_numpy(dtype=float, copy=False)
                    if not np.isfinite(values).all():
                        continue
                    candidates.append(
                        (
                            header_row,
                            data_column,
                            data_row,
                            label_column,
                            block,
                            horizontal_raw,
                        )
                    )

    if not candidates:
        raise ValueError(
            f"could not locate a {sector_count}x{sector_count} presentation-style IO block"
        )

    # Prefer the earliest/left-most structurally valid block.  On PSA transaction sheets
    # this selects intermediate demand, before totals/final-demand columns.
    candidates.sort(key=lambda item: item[:4])
    _, _, _, _, block, labels = candidates[0]
    block = block.copy()
    block.index = labels
    block.columns = labels
    return block


def _canonicalize_expected_order(frame: pd.DataFrame, sector_count: int | None) -> pd.DataFrame:
    """Map a published 16/80-sector ordering to SIGMA's ``01..N`` identifiers.

    A clean machine table that already carries canonical IDs is left unchanged.  A published
    table often uses PSIC ranges or long descriptions on both axes; when the caller has
    explicitly selected IO16 or IO80, positional order is the interoperability contract.
    """
    if sector_count is None:
        return frame
    if frame.shape != (sector_count, sector_count):
        raise ValueError(
            f"selected IO resolution expects {sector_count} sectors, but the transaction "
            f"block is {frame.shape[0]}x{frame.shape[1]}"
        )
    canonical = [f"{position:02d}" for position in range(1, sector_count + 1)]
    if list(frame.index) == canonical and list(frame.columns) == canonical:
        return frame
    out = frame.copy()
    out.index = canonical
    out.columns = canonical
    return out


def read_io_table(
    path: str | Path,
    sheet_name: str | int = 0,
    expected_sector_count: int | None = None,
) -> pd.DataFrame:
    """Read and validate a square IO transaction table.

    The first column is the row-sector identifier and the remaining column names are
    column-sector identifiers.  Rows are suppliers and columns are users, so a positive
    entry ``z[i,j]`` produces a directed edge ``i -> j``.

    SIGMA intentionally requires the normalized row and column sector *sets* to match
    exactly.  Silently dropping unmatched sectors can change the MWAS objective and therefore
    the selected IO DAG, without an obvious error in the final spatial network.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IO table does not exist: {path}")

    suffix = path.suffix.lower()
    if expected_sector_count is not None and expected_sector_count not in {16, 80}:
        raise ValueError("expected_sector_count must be None, 16, or 80")

    if suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
        used_presentation = False
        try:
            validated = _validated_transaction_frame(frame)
        except ValueError as strict_error:
            # Formatted publication workbooks are not machine-square tables.  Only invoke
            # structural extraction when a concrete IO16/IO80 size is known; otherwise the
            # strict error is more informative than guessing a matrix size.
            if expected_sector_count is None:
                raise strict_error
            raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
            extracted = _extract_presentation_block(raw, expected_sector_count)
            validated = _validated_transaction_frame(extracted)
            used_presentation = True
        validated = _canonicalize_expected_order(validated, expected_sector_count)
        validated.attrs["source_layout"] = (
            "presentation-published-order" if used_presentation else "clean-square"
        )
        return validated
    elif suffix == ".xls":
        raise ValueError(
            "legacy .xls input is not supported by sigma-engine's minimal dependency set; "
            "save the workbook as .xlsx or CSV"
        )
    elif suffix == ".csv":
        frame = pd.read_csv(path, index_col=0)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        # A Parquet table may have been written without preserving a semantic index.  In
        # that common case, interpret the first column exactly as CSV/Excel do.
        if isinstance(frame.index, pd.RangeIndex):
            if frame.shape[1] < 2:
                raise ValueError("IO Parquet table must include a sector-id column")
            frame = frame.set_index(frame.columns[0])
    else:
        raise ValueError(f"unsupported IO table format: {suffix or '<no suffix>'}")

    validated = _validated_transaction_frame(frame)
    validated = _canonicalize_expected_order(validated, expected_sector_count)
    validated.attrs["source_layout"] = "canonical-or-clean-square"
    return validated


def io_network(table: pd.DataFrame) -> nx.DiGraph:
    """Convert a validated transaction table into a weighted directed IO graph."""
    if list(table.index) != list(table.columns):
        raise ValueError("IO table axes must be aligned before graph construction")

    values = table.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("IO table values must be finite and non-negative")

    graph = nx.DiGraph()
    graph.add_nodes_from(str(sector) for sector in table.index)

    sectors = [str(sector) for sector in table.index]
    for row, source in enumerate(sectors):
        for column, target in enumerate(sectors):
            weight = float(values[row, column])
            # Positive self-use remains in the source IO graph for faithful accounting.
            # No self-loop can belong to a DAG, so every MWAS method necessarily excludes
            # it from the retained acyclic subgraph while still counting its weight as removed.
            if weight <= 0:
                continue
            graph.add_edge(source, target, weight=weight)
    return graph




def _validate_weighted_digraph(graph: nx.DiGraph) -> None:
    """Validate the non-negative directed graph required by both MWAS methods."""
    if not graph.is_directed():
        raise ValueError("MWAS requires a directed graph")
    for source, target, data in graph.edges(data=True):
        if "weight" not in data:
            raise ValueError(f"IO edge {source!r}->{target!r} is missing weight")
        weight = float(data["weight"])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(
                f"IO edge {source!r}->{target!r} has invalid non-negative finite weight: "
                f"{data['weight']!r}"
            )


def _graph_weight(graph: nx.DiGraph) -> float:
    return float(sum(float(data["weight"]) for _, _, data in graph.edges(data=True)))


def _candidate_edges(graph: nx.DiGraph) -> list[tuple[object, object, float]]:
    """Return deterministic positive, non-self IO edges eligible for a DAG."""
    return sorted(
        (
            (source, target, float(data["weight"]))
            for source, target, data in graph.edges(data=True)
            if source != target and float(data["weight"]) > 0.0
        ),
        key=lambda edge: (str(edge[0]), str(edge[1])),
    )


def _result(
    source: nx.DiGraph,
    dag: nx.DiGraph,
    *,
    method: str,
    exact: bool,
    optimal: bool,
    solver_status: str | None,
) -> MWASResult:
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError(f"{method} MWAS produced a cyclic graph")
    source_weight = _graph_weight(source)
    retained_weight = _graph_weight(dag)
    removed_weight = source_weight - retained_weight
    scale = max(1.0, abs(source_weight))
    if removed_weight < -1e-10 * scale:
        raise RuntimeError("MWAS retained more total weight than exists in the source graph")
    self_loop_weight = float(
        sum(
            float(data["weight"])
            for source_node, target_node, data in source.edges(data=True)
            if source_node == target_node
        )
    )
    return MWASResult(
        graph=dag,
        method=method,
        exact=exact,
        optimal=optimal,
        solver_status=solver_status,
        source_weight=source_weight,
        retained_weight=retained_weight,
        removed_weight=max(0.0, float(removed_weight)),
        self_loop_weight=self_loop_weight,
        source_edge_count=int(source.number_of_edges()),
        retained_edge_count=int(dag.number_of_edges()),
        removed_edge_count=int(source.number_of_edges() - dag.number_of_edges()),
    )


def fast_mwas(graph: nx.DiGraph) -> MWASResult:
    """Return SIGMA's deterministic fast MWAS heuristic.

    Edges are considered in descending weight order, with source/target string
    order used only to make equal-weight ties deterministic.  An edge is accepted
    iff there is not already a path from its target back to its source.  Because
    the partial result starts acyclic, this single reachability test is exactly the
    condition for preserving acyclicity.

    This produces a *maximal* weighted acyclic subgraph quickly, but not
    necessarily the globally maximum-weight one.  Hassin & Rubinstein (1994)
    provide general algorithmic background for maximum acyclic subgraph
    approximation; no approximation ratio is claimed for this specific greedy
    ordering.
    """
    _validate_weighted_digraph(graph)
    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes(data=True))
    edges = sorted(
        _candidate_edges(graph),
        key=lambda edge: (-edge[2], str(edge[0]), str(edge[1])),
    )
    for source, target, weight in edges:
        if nx.has_path(dag, target, source):
            continue
        dag.add_edge(source, target, weight=weight)
    return _result(
        graph,
        dag,
        method="fast_greedy",
        exact=False,
        optimal=False,
        solver_status=None,
    )


def exact_mwas(graph: nx.DiGraph) -> MWASResult:
    """Solve weighted MWAS exactly with a topological-rank MILP.

    For every eligible edge ``e=(u,v)`` there is a binary keep variable ``x_e``
    and every node receives an integer rank in ``[0, n-1]``.  With ``M=n`` the
    constraint

        rank(u) - rank(v) + M*x_e <= M - 1

    is inactive when ``x_e=0`` and becomes ``rank(u)+1 <= rank(v)`` when the edge
    is retained.  Maximizing ``sum(weight_e*x_e)`` is therefore the exact MWAS
    objective.  SciPy's ``milp`` delegates to HiGHS and is requested with zero
    relative MIP gap.  Failure to prove an optimum raises; exact mode never
    silently falls back to the heuristic.
    """
    _validate_weighted_digraph(graph)
    nodes = sorted(graph.nodes, key=str)
    edges = _candidate_edges(graph)
    n = len(nodes)
    m = len(edges)
    if m == 0:
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes(data=True))
        return _result(
            graph,
            dag,
            method="exact_milp",
            exact=True,
            optimal=True,
            solver_status="trivial",
        )

    node_index = {node: idx for idx, node in enumerate(nodes)}
    weights = np.asarray([edge[2] for edge in edges], dtype=float)
    # Positive scaling preserves the exact optimizer while keeping the objective
    # numerically well-conditioned across IO tables measured in different units.
    weight_scale = float(max(1.0, np.max(weights)))
    objective = np.zeros(m + n, dtype=float)
    objective[:m] = -(weights / weight_scale)

    row = np.repeat(np.arange(m, dtype=np.int64), 3)
    col = np.empty(3 * m, dtype=np.int64)
    data = np.empty(3 * m, dtype=float)
    for edge_idx, (source, target, _weight) in enumerate(edges):
        base = 3 * edge_idx
        col[base] = edge_idx
        data[base] = float(n)
        col[base + 1] = m + node_index[source]
        data[base + 1] = 1.0
        col[base + 2] = m + node_index[target]
        data[base + 2] = -1.0
    matrix = coo_matrix((data, (row, col)), shape=(m, m + n)).tocsr()
    constraints = LinearConstraint(
        matrix,
        lb=np.full(m, -np.inf, dtype=float),
        ub=np.full(m, float(n - 1), dtype=float),
    )
    lower = np.zeros(m + n, dtype=float)
    upper = np.concatenate(
        [np.ones(m, dtype=float), np.full(n, float(max(0, n - 1)), dtype=float)]
    )
    result = milp(
        c=objective,
        integrality=np.ones(m + n, dtype=np.int8),
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"mip_rel_gap": 0.0, "presolve": True},
    )
    if not bool(result.success) or int(result.status) != 0 or result.x is None:
        raise RuntimeError(
            "exact MWAS did not return a proven optimum: "
            f"status={result.status}, message={result.message}"
        )

    dag = nx.DiGraph()
    dag.add_nodes_from(graph.nodes(data=True))
    selected = np.asarray(result.x[:m]) > 0.5
    for keep, (source, target, weight) in zip(selected, edges, strict=True):
        if keep:
            dag.add_edge(source, target, weight=weight)
    return _result(
        graph,
        dag,
        method="exact_milp",
        exact=True,
        optimal=True,
        solver_status=str(result.message),
    )


def solve_mwas(graph: nx.DiGraph, *, method: str = "fast") -> MWASResult:
    """Solve/approximate MWAS with ``fast`` (default) or ``exact`` mode."""
    if method == "fast":
        return fast_mwas(graph)
    if method == "exact":
        return exact_mwas(graph)
    raise ValueError("MWAS method must be 'fast' or 'exact'")


def maximum_weight_acyclic_subgraph(
    graph: nx.DiGraph,
    *,
    exact: bool = False,
) -> nx.DiGraph:
    """Compatibility API returning only the MWAS graph.

    ``exact=False`` is deliberately the default in SIGMA 0.2.7 and later.
    Use :func:`solve_mwas` when optimization metadata is also needed.
    """
    return solve_mwas(graph, method="exact" if exact else "fast").graph
