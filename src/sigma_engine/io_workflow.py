"""Step 5 contracts for the revised SIGMA workflow.

Transactions select the acyclic edge set through MWAS. Technical coefficients are
then attached to *that surviving edge set* and become the downstream DAG weights.
When a full transaction workbook supplies sector total output, SIGMA derives the
technical coefficients exactly as ``A_ij = Z_ij / x_j``. An explicit A matrix remains
available as an override. The two roles are intentionally never conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .io_dag import MWASResult, io_network, read_io_table, solve_mwas


@dataclass(frozen=True)
class IODAGResult:
    """Reweighted IO DAG plus the underlying MWAS result."""

    graph: nx.DiGraph
    mwas: MWASResult
    rows: pd.DataFrame


def _display_token(value: object) -> str | None:
    """Return a whitespace-normalized, case-insensitive spreadsheet label."""
    if value is None or pd.isna(value):
        return None
    token = " ".join(str(value).split()).strip().casefold()
    return token or None


def _locate_presentation_transaction_block(
    raw: pd.DataFrame,
    sector_count: int,
) -> tuple[int, int, int, int]:
    """Locate the same supplier-user block selected by ``read_io_table``.

    The selection rule intentionally mirrors ``io_dag._extract_presentation_block``:
    matching ordered sector labels must appear horizontally above the numeric block and
    vertically to its left, and the earliest/left-most structurally valid block wins.
    Returning its coordinates lets the total-output reader align ``x`` with the same rows.
    """
    if sector_count < 2:
        raise ValueError("sector_count must be >= 2")
    rows, columns = raw.shape
    if rows < sector_count or columns < sector_count:
        raise ValueError(
            f"worksheet is too small to contain a {sector_count}x{sector_count} transaction block"
        )

    candidates: list[tuple[int, int, int, int]] = []
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
                    if np.isfinite(values).all():
                        candidates.append(
                            (header_row, data_column, data_row, label_column)
                        )

    if not candidates:
        raise ValueError(
            f"could not locate a {sector_count}x{sector_count} presentation-style IO block"
        )
    candidates.sort()
    return candidates[0]


def read_total_output_vector(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    sector_ids: list[str] | pd.Index,
) -> pd.Series:
    """Read sector gross output ``x`` from a full presentation-style IO workbook.

    Automatic derivation deliberately requires an explicit ``Total Output`` column in
    the same worksheet as the transaction block. The square packaged/intermediate-only
    matrices do not contain enough information and are never normalized by their column
    sums as a substitute.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IO transaction table does not exist: {path}")
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(
            "automatic technical-coefficient derivation currently requires an .xlsx/.xlsm "
            "transaction workbook containing an explicit Total Output column"
        )

    ids = [str(value) for value in sector_ids]
    sector_count = len(ids)
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row, data_column, data_row, _ = _locate_presentation_transaction_block(
        raw, sector_count
    )

    header_candidates: list[int] = []
    for row in range(header_row, data_row):
        for column in range(data_column + sector_count, raw.shape[1]):
            if _display_token(raw.iat[row, column]) == "total output":
                header_candidates.append(column)
    header_candidates = sorted(set(header_candidates))

    valid: list[tuple[int, np.ndarray]] = []
    for column in header_candidates:
        values = pd.to_numeric(
            raw.iloc[data_row : data_row + sector_count, column],
            errors="coerce",
        ).to_numpy(dtype=float)
        if np.isfinite(values).all() and np.all(values > 0):
            valid.append((column, values))

    if not valid:
        raise ValueError(
            "transaction workbook does not expose a usable Total Output column aligned "
            "with the selected IO sector rows; supply the full transaction workbook or "
            "provide --technical-coefficients explicitly"
        )
    if len(valid) > 1:
        columns = [column for column, _ in valid]
        raise ValueError(
            "transaction workbook has multiple usable Total Output columns aligned with "
            f"the IO block at columns {columns}; derivation is ambiguous"
        )

    column, values = valid[0]
    output = pd.Series(values, index=ids, dtype=float, name="total_output")
    output.attrs["source_column_zero_based"] = int(column)
    output.attrs["source_sheet"] = sheet_name
    return output


def derive_technical_coefficients(
    transactions: pd.DataFrame,
    total_output: pd.Series,
) -> pd.DataFrame:
    """Derive ``A_ij = Z_ij / x_j`` using gross output for each user sector."""
    z = transactions.copy()
    if list(z.index) != list(z.columns):
        raise ValueError("transaction matrix axes must be aligned")
    ids = [str(value) for value in z.index]
    z.index = ids
    z.columns = [str(value) for value in z.columns]

    x = pd.Series(total_output, copy=True, dtype=float)
    x.index = [str(value) for value in x.index]
    if x.index.duplicated().any():
        raise ValueError("total-output sector identifiers must be unique")
    if set(x.index) != set(ids):
        only_z = sorted(set(ids) - set(x.index))
        only_x = sorted(set(x.index) - set(ids))
        raise ValueError(
            "transactions and total output must contain exactly the same sectors; "
            f"only in transactions={only_z[:10]}, only in total output={only_x[:10]}"
        )
    x = x.loc[ids]
    values = x.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("total output must be finite and strictly positive for every sector")

    z_values = z.to_numpy(dtype=float, copy=False)
    if not np.isfinite(z_values).all() or np.any(z_values < 0):
        raise ValueError("transactions must be finite and non-negative")

    coefficients = z.astype(float).div(x, axis="columns")
    a_values = coefficients.to_numpy(dtype=float, copy=False)
    if not np.isfinite(a_values).all() or np.any(a_values < 0):
        raise RuntimeError("derived technical coefficients are not finite and non-negative")
    coefficients.attrs["technical_coefficient_source"] = (
        "derived_from_transaction_total_output"
    )
    return coefficients


def read_technical_coefficients(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    expected_sector_count: int | None = None,
) -> pd.DataFrame:
    """Read an explicit square technical-coefficient matrix override."""
    coefficients = read_io_table(
        path,
        sheet_name=sheet_name,
        expected_sector_count=expected_sector_count,
    )
    coefficients.attrs["technical_coefficient_source"] = "explicit_matrix"
    return coefficients


def _validate_matrix_pair(
    transactions: pd.DataFrame,
    technical_coefficients: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align Z and A without silently dropping sectors."""
    z = transactions.copy()
    a = technical_coefficients.copy()

    if list(z.index) != list(z.columns):
        raise ValueError("transaction matrix axes must be aligned")
    if list(a.index) != list(a.columns):
        raise ValueError("technical-coefficient matrix axes must be aligned")

    z_ids = [str(v) for v in z.index]
    a_ids = [str(v) for v in a.index]
    if set(z_ids) != set(a_ids):
        only_z = sorted(set(z_ids) - set(a_ids))
        only_a = sorted(set(a_ids) - set(z_ids))
        raise ValueError(
            "transactions and technical coefficients must contain exactly the same sectors; "
            f"only in transactions={only_z[:10]}, only in coefficients={only_a[:10]}"
        )

    z.index = z_ids
    z.columns = [str(v) for v in z.columns]
    a.index = a_ids
    a.columns = [str(v) for v in a.columns]
    a = a.loc[z_ids, z_ids]

    for name, frame in (("transactions", z), ("technical coefficients", a)):
        values = frame.to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contain missing or non-finite values")
        if (values < 0).any():
            raise ValueError(f"{name} must be non-negative")
    return z.astype(float), a.astype(float)


def reweight_surviving_graph(
    surviving_graph: nx.DiGraph,
    transactions: pd.DataFrame,
    technical_coefficients: pd.DataFrame,
    *,
    mwas_method: str,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Attach transaction audit values and A weights to an already-selected DAG."""
    z, a = _validate_matrix_pair(transactions, technical_coefficients)
    if not nx.is_directed_acyclic_graph(surviving_graph):
        raise ValueError("MWAS surviving graph must be acyclic before reweighting")

    dag = nx.DiGraph()
    dag.add_nodes_from(str(node) for node in surviving_graph.nodes)
    rows: list[dict[str, object]] = []
    for source, target in sorted(
        surviving_graph.edges(), key=lambda edge: (str(edge[0]), str(edge[1]))
    ):
        source_text, target_text = str(source), str(target)
        if source_text not in z.index or target_text not in z.columns:
            raise ValueError(
                f"MWAS edge {source_text!r}->{target_text!r} is absent from the IO matrices"
            )
        transaction_value = float(z.loc[source_text, target_text])
        technical_coefficient = float(a.loc[source_text, target_text])
        if transaction_value <= 0:
            raise RuntimeError(
                f"MWAS retained non-positive transaction edge {source_text!r}->{target_text!r}"
            )
        if technical_coefficient <= 0:
            raise ValueError(
                "a surviving positive transaction edge must have a positive technical "
                f"coefficient; got A[{source_text},{target_text}]={technical_coefficient}"
            )
        dag.add_edge(
            source_text,
            target_text,
            weight=technical_coefficient,
            transaction_value=transaction_value,
            technical_coefficient=technical_coefficient,
            mwas_method=mwas_method,
        )
        rows.append(
            {
                "source_type": source_text,
                "target_type": target_text,
                "transaction_value": transaction_value,
                "technical_coefficient": technical_coefficient,
                "mwas_method": mwas_method,
            }
        )

    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("technical-coefficient reweighting changed DAG acyclicity")
    return dag, pd.DataFrame(
        rows,
        columns=[
            "source_type",
            "target_type",
            "transaction_value",
            "technical_coefficient",
            "mwas_method",
        ],
    )


def build_io_dag(
    transactions: pd.DataFrame,
    technical_coefficients: pd.DataFrame,
    *,
    method: str = "fast",
) -> IODAGResult:
    """Run MWAS on Z, then reweight surviving edges with A."""
    z, a = _validate_matrix_pair(transactions, technical_coefficients)
    source = io_network(z)
    mwas = solve_mwas(source, method=method)
    dag, rows = reweight_surviving_graph(
        mwas.graph,
        z,
        a,
        mwas_method=method,
    )
    return IODAGResult(graph=dag, mwas=mwas, rows=rows)
