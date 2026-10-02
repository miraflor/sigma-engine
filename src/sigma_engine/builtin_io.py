"""Built-in Philippine input-output tables and explicit custom overrides.

Normal SIGMA runs are self-contained: the package ships the normalized 2018 PSA
IO80 and IO16 intermediate transaction matrices together with the corresponding
sector gross-output vectors.  ``Z`` therefore selects MWAS edges and ``A`` can be
derived internally as ``A_ij = Z_ij / x_j`` without an external economic file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .io_dag import read_io_table

PSA_RELEASE_URL = "https://psa.gov.ph/content/psa-releases-2018-input-output-tables"


@dataclass(frozen=True)
class IOInputInfo:
    """Provenance for the transaction matrix used in one engine run."""

    source_id: str
    source_kind: Literal["builtin", "override"]
    source_path: str | None
    source_url: str | None
    reference_year: int | None
    resource_sha256: str | None


_BUILTINS: dict[str, tuple[str, int, str, str]] = {
    "io80": (
        "psa_2018_io80.csv",
        80,
        "psa-2018-io80-transaction",
        "9335f4de98d6b11205b431f8702f26a2c932d641fdd911ce7ddcf08f604840a1",
    ),
    "io16": (
        "psa_2018_io16.csv",
        16,
        "psa-2018-io16-transaction",
        "f5a2105fc67d2c992eb86531b310b0ce7eff3046a981da88570cea196a49ba76",
    ),
}

_TOTAL_OUTPUTS: dict[str, tuple[str, int, str]] = {
    "io80": (
        "psa_2018_io80_total_output.csv",
        80,
        "7159ef3cdc7368d949b4846e21927ddba3c5d116c7cceafcd9efd48273aca061",
    ),
    "io16": (
        "psa_2018_io16_total_output.csv",
        16,
        "8b5653e65002fbedcb7602793da044c4a13cd9664bc0e7f8d7d41ebe9f53e9cf",
    ),
}


def _canonical_text_sha256(payload: bytes) -> str:
    """Hash a text resource independently of checkout newline convention."""
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _load_builtin(classification: str) -> tuple[pd.DataFrame, IOInputInfo]:
    """Load one packaged canonical transaction matrix and verify its contents."""
    try:
        filename, sector_count, source_id, expected_sha256 = _BUILTINS[classification]
    except KeyError as exc:
        raise ValueError("classification must be 'io80' or 'io16'") from exc

    resource = files("sigma_engine.resources").joinpath(filename)
    payload = resource.read_bytes()
    actual_sha256 = _canonical_text_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"bundled {classification} transaction resource failed integrity check: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    with as_file(resource) as materialized:
        table = read_io_table(materialized, expected_sector_count=sector_count)

    table.attrs["source_layout"] = "builtin-canonical"
    table.attrs["source_id"] = source_id
    return table, IOInputInfo(
        source_id=source_id,
        source_kind="builtin",
        source_path=None,
        source_url=PSA_RELEASE_URL,
        reference_year=2018,
        resource_sha256=actual_sha256,
    )


def load_builtin_total_output(classification: str) -> pd.Series:
    """Load the packaged gross-output vector aligned to the built-in IO matrix."""
    try:
        filename, sector_count, expected_sha256 = _TOTAL_OUTPUTS[classification]
    except KeyError as exc:
        raise ValueError("classification must be 'io80' or 'io16'") from exc

    resource = files("sigma_engine.resources").joinpath(filename)
    payload = resource.read_bytes()
    actual_sha256 = _canonical_text_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"bundled {classification} total-output resource failed integrity check: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    with as_file(resource) as materialized:
        frame = pd.read_csv(materialized, dtype={"sector": str})

    if list(frame.columns) != ["sector", "total_output"]:
        raise RuntimeError(
            f"bundled {classification} total-output resource has an unexpected schema"
        )
    canonical = [f"{position:02d}" for position in range(1, sector_count + 1)]
    sectors = frame["sector"].astype(str).str.zfill(2).tolist()
    if sectors != canonical:
        raise RuntimeError(
            f"bundled {classification} total-output sectors are not canonical 01..{sector_count}"
        )

    values = pd.to_numeric(frame["total_output"], errors="coerce").to_numpy(float)
    if len(values) != sector_count or not np.isfinite(values).all() or np.any(values <= 0):
        raise RuntimeError(
            f"bundled {classification} total output must contain {sector_count} finite "
            "strictly-positive values"
        )

    output = pd.Series(values, index=canonical, dtype=float, name="total_output")
    output.attrs["source_id"] = f"psa-2018-{classification}-total-output"
    output.attrs["source_kind"] = "builtin"
    output.attrs["source_url"] = PSA_RELEASE_URL
    output.attrs["reference_year"] = 2018
    output.attrs["resource_sha256"] = actual_sha256
    output.attrs["resource_filename"] = filename
    return output


def load_engine_io_table(
    classification: str,
    *,
    override_path: str | Path | None = None,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, IOInputInfo]:
    """Resolve the transaction matrix for one engine run."""
    if classification not in _BUILTINS:
        raise ValueError("classification must be 'io80' or 'io16'")

    if override_path is None:
        return _load_builtin(classification)

    path = Path(override_path).expanduser().resolve()
    sector_count = _BUILTINS[classification][1]
    table = read_io_table(path, sheet_name, expected_sector_count=sector_count)
    table.attrs["source_id"] = "custom-override"
    return table, IOInputInfo(
        source_id="custom-override",
        source_kind="override",
        source_path=str(path),
        source_url=None,
        reference_year=None,
        resource_sha256=None,
    )
