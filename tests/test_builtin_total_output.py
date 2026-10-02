import math

import numpy as np

from sigma_engine.builtin_io import load_builtin_total_output


def test_builtin_io80_total_output_is_canonical_and_positive():
    output = load_builtin_total_output("io80")

    assert len(output) == 80
    assert list(output.index) == [f"{i:02d}" for i in range(1, 81)]
    assert np.isfinite(output.to_numpy()).all()
    assert (output.to_numpy() > 0).all()
    assert output.attrs["source_kind"] == "builtin"
    assert output.attrs["reference_year"] == 2018
    assert output.attrs["resource_sha256"] == (
        "7159ef3cdc7368d949b4846e21927ddba3c5d116c7cceafcd9efd48273aca061"
    )
    assert math.isclose(output.loc["01"], 501502.445511913, rel_tol=1e-14)
    assert math.isclose(output.loc["80"], 265745.971793665, rel_tol=1e-14)


def test_builtin_io16_total_output_is_canonical_and_positive():
    output = load_builtin_total_output("io16")

    assert len(output) == 16
    assert list(output.index) == [f"{i:02d}" for i in range(1, 17)]
    assert np.isfinite(output.to_numpy()).all()
    assert (output.to_numpy() > 0).all()
    assert output.attrs["resource_sha256"] == (
        "8b5653e65002fbedcb7602793da044c4a13cd9664bc0e7f8d7d41ebe9f53e9cf"
    )
    assert math.isclose(output.loc["01"], 3460981.65433775, rel_tol=1e-14)
    assert math.isclose(output.loc["16"], 592941.898663591, rel_tol=1e-14)
