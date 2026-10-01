"""Built-in Philippine input-output tables and explicit custom overrides.

The engine owns its economic-network specification.  Normal runs therefore select the
bundled 2018 PSA transaction matrix that matches ``classification``; callers do not need
to pass an IO workbook.  A custom table remains available as an explicit experimental
override rather than part of the ordinary workflow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal

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


def _canonical_text_sha256(payload: bytes) -> str:
    """Hash a text resource independently of checkout newline convention.

    Git may materialize text files with CRLF on Windows and LF on Unix-like systems.
    The IO matrix values are unchanged by that translation, so the integrity check
    canonicalizes all text newlines to LF before hashing. This still detects any
    substantive byte change while avoiding false failures caused solely by Git EOL
    conversion.
    """
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

    # ``as_file`` works both for an editable source tree and for a wheel whose resources
    # may not have a permanent filesystem path.
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


def load_engine_io_table(
    classification: str,
    *,
    override_path: str | Path | None = None,
    sheet_name: str | int = 0,
) -> tuple[pd.DataFrame, IOInputInfo]:
    """Resolve the IO matrix for one engine run.

    Built-in IO80 is selected by the engine's default ``classification='io80'``.  IO16
    selects the bundled 16-sector aggregation.  ``override_path`` is intentionally explicit:
    supplying it replaces the built-in matrix for custom/experimental work only.
    """
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
