# SIGMA Engine built-in PSA economic-input patch r6

Baseline: `411ad11d9363837bd58d17aa063a285bc5ad7ad1`.

## Purpose

Make the normal IO80/IO16 workflow fully self-contained while preserving the
revised Step-5 distinction:

- transaction values `Z` select MWAS edges;
- technical coefficients `A` weight surviving edges;
- `A_ij = Z_ij / x_j`, with `x_j` equal to published total gross output.

## Bundled economic resources

| Classification | Resource | Canonical SHA-256 |
|---|---|---|
| IO80 | `src/sigma_engine/resources/psa_2018_io80.csv` | `9335f4de98d6b11205b431f8702f26a2c932d641fdd911ce7ddcf08f604840a1` |
| IO80 | `src/sigma_engine/resources/psa_2018_io80_total_output.csv` | `7159ef3cdc7368d949b4846e21927ddba3c5d116c7cceafcd9efd48273aca061` |
| IO16 | `src/sigma_engine/resources/psa_2018_io16.csv` | `f5a2105fc67d2c992eb86531b310b0ce7eff3046a981da88570cea196a49ba76` |
| IO16 | `src/sigma_engine/resources/psa_2018_io16_total_output.csv` | `8b5653e65002fbedcb7602793da044c4a13cd9664bc0e7f8d7d41ebe9f53e9cf` |

Text hashes are checked after canonicalizing newline convention, so Git's LF/CRLF
checkout behavior on Windows does not cause false integrity failures.

## Files in the overlay

- `APPLY.txt`
- `CHANGELOG.md`
- `CHECKSUMS.sha256`
- `OVERLAY_MANIFEST.md`
- `README.md`
- `docs/IMPLEMENTATION_NOTES.md`
- `docs/SIGMA_REVISED_WORKFLOW.md`
- `docs/TESTING.md`
- `src/sigma_engine/builtin_io.py`
- `src/sigma_engine/cli.py`
- `src/sigma_engine/io_workflow.py`
- `src/sigma_engine/pipeline.py`
- `src/sigma_engine/resources/NOTICE.txt`
- `src/sigma_engine/resources/psa_2018_io16.csv`
- `src/sigma_engine/resources/psa_2018_io16_total_output.csv`
- `src/sigma_engine/resources/psa_2018_io80.csv`
- `src/sigma_engine/resources/psa_2018_io80_total_output.csv`
- `tests/test_builtin_total_output.py`
- `tests/test_cli.py`
- `tests/test_io_input_resolution.py`
- `tests/test_io_workflow_revised.py`

## Default economic-input behavior

With no economic-file arguments:

1. `classification=io80` loads the internal 80-industry `Z` and `x` resources.
2. `classification=io16` loads the internal 16-industry `Z` and `x` resources.
3. SIGMA derives `A` column-wise from `Z/x`.
4. Fast MWAS selects the acyclic edge set from transaction weights in `Z`.
5. Only surviving MWAS edges are reweighted with `A`.

`--transactions` and `--technical-coefficients` remain optional custom overrides.
A full custom transaction workbook may provide its own explicit `Total Output` column.
A square intermediate-only custom matrix cannot manufacture `x` from column sums.

## Source

Philippine Statistics Authority, 2018 Benchmark Input-Output Accounts of the
Philippines, released 9 December 2021, Reference No. 2021-510. The PSA release
publishes transaction tables at 16x16, 80x80, and 240x240 resolution.
