from pathlib import Path

import pandas as pd
import pytest
from pytest import fixture

from h2integrate.core.file_utils import load_yaml
from h2integrate.tools.profast_tools import (
    run_profast,
    convert_pf_to_dict,
    make_price_breakdown,
    create_and_populate_profast,
    format_profast_price_breakdown_per_year,
)


@fixture
def profast_config():
    config_fpath = Path(__file__).parent / "profast_config.yaml"
    pf_config = load_yaml(config_fpath)

    return pf_config


@pytest.mark.regression
def test_lco_breakdown(profast_config, subtests):
    pf = create_and_populate_profast(profast_config)
    sol, _summary, price_breakdown = run_profast(pf)
    _full_price_breakdown, lco_check = make_price_breakdown(price_breakdown, profast_config)

    lcoe_initial = float(sol["price"] * 1e3)

    lco_from_breakdown = lco_check * 1e3
    with subtests.test("Breakdown LCOE matches actual LCOE"):
        assert pytest.approx(lco_from_breakdown, rel=1e-6) == lcoe_initial

    with subtests.test(
        "Breakdown LCOE with config created from profast object matches actual LCOE"
    ):
        pf_config_dict = convert_pf_to_dict(pf)
        _full_price_breakdown, lco_check = make_price_breakdown(price_breakdown, pf_config_dict)

        assert pytest.approx(lco_check * 1e3, rel=1e-6) == lcoe_initial


@pytest.mark.unit
def test_price_breakdown_preserves_rows_with_unequal_cash_flow_lengths():
    breakdown = pd.DataFrame(
        {
            "Type": ["Operating revenue", "Depreciation expense", "Financing cash outflow"],
            "Name": ["sales", "depreciation", "debt"],
            "Amount": [[1.0, 2.0], [9.0, 9.0], [3.0, 4.0, 5.0]],
            "NPV": [10.0, 20.0, 30.0],
        },
        index=[2, 4, 7],
    )
    result = format_profast_price_breakdown_per_year(breakdown)
    assert result.index.tolist() == [2, 7]
    assert result.loc[2, "Name"] == "sales"
    assert result.loc[2, "Year 1 Amount"] == 2.0
    assert pd.isna(result.loc[2, "Year 2 Amount"])
    assert result.loc[7, "Name"] == "debt"
    assert result.loc[7, "Year 2 Amount"] == 5.0
    assert result.loc[7, "NPV"] == 30.0
