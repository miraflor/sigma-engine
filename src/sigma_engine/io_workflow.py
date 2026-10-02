"""Step 5 contracts for the revised SIGMA workflow.

Transactions select the acyclic edge set through MWAS.  Technical coefficients are
then attached to *that surviving edge set* and become the downstream DAG weights.
The two matrices are intentionally never conflated.
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


def read_technical_coefficients(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    expected_sector_count: int | None = None,
) -> pd.DataFrame:
    """Read a square technical-coefficient matrix using the IO table parser.

    The parser's structural validation is useful for both Z and A.  No attempt is
    made to derive A from Z because total sector output is not recoverable from an
    intermediate-transactions block alone.
    """
    return read_io_table(
        path,
        sheet_name=sheet_name,
        expected_sector_count=expected_sector_count,
    )


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
