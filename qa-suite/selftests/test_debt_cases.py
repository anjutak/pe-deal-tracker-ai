"""
Tests for the 6 debt cases in golden_dataset.json: recompute each expected
answer straight from deals_with_debt.csv, written separately from
add_debt_cases.py so a bug in one is caught by the other.

Runs offline - no Langflow, no API keys.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ROWS = list(csv.DictReader(open(ROOT / "deals_with_debt.csv", encoding="utf-8")))
CASES = {c["id"]: c for c in json.loads((ROOT / "golden_dataset.json").read_text(encoding="utf-8"))["cases"]}
DEBT_IDS = [f"debt_0{i}" for i in range(1, 7)]


def by_name(name):
    return next(r for r in ROWS if r["CompanyName"] == name)


def required(case_id):
    return {g[0] for g in CASES[case_id]["must_contain"]}


def forbidden(case_id):
    return {g[0] for g in CASES[case_id]["must_not_contain"]}


def accepts(case_id, text):
    """True if `text` would pass every must_contain group of the case."""
    return all(any(alt.lower() in text.lower() for alt in g) for g in CASES[case_id]["must_contain"])


# ---- The data file itself ------------------------------------------------------
def test_debt_file_keeps_the_original_50_deals_and_columns():
    original = list(csv.DictReader(open(ROOT / "fake_investment_pipeline_unique_companies.csv", encoding="utf-8")))
    assert len(ROWS) == 50
    for old, new in zip(original, ROWS):
        assert all(new[k] == v for k, v in old.items()), f"{old['CompanyName']} changed"


def test_every_deal_has_consistent_debt_fields():
    for r in ROWS:
        if r["DebtType"] == "None":
            assert r["DebtAmountUSD"] == "0" and r["RateType"] == "" and r["MaturityYear"] == ""
        elif r["RateType"] == "Fixed":
            assert r["FixedRatePct"] and not r["SpreadOverSOFRbps"]
        else:
            assert r["RateType"] == "Floating" and r["SpreadOverSOFRbps"] and not r["FixedRatePct"]


# ---- Each debt case ------------------------------------------------------------
@pytest.mark.parametrize("case_id", DEBT_IDS)
def test_debt_case_exists(case_id):
    assert case_id in CASES


def test_debt_01_spread():
    r = by_name("Pioneer Dynamics")
    assert r["RateType"] == "Floating" and r["SpreadOverSOFRbps"] == "475"
    assert accepts("debt_01", "SOFR + 475 bps") and accepts("debt_01", "SOFR plus 4.75%")


def test_debt_02_maturing_before_2030():
    expected = {r["CompanyName"] for r in ROWS if r["MaturityYear"] and int(r["MaturityYear"]) < 2030}
    in_2030 = {r["CompanyName"] for r in ROWS if r["MaturityYear"] == "2030"}
    assert required("debt_02") == expected
    assert forbidden("debt_02") == in_2030 and in_2030, "boundary trap needs 2030 deals"


def test_debt_03_fixed_rate():
    assert required("debt_03") == {r["CompanyName"] for r in ROWS if r["RateType"] == "Fixed"}
    assert forbidden("debt_03") == {r["CompanyName"] for r in ROWS if r["RateType"] == "Floating"}


def test_debt_04_total_debt():
    total = sum(int(r["DebtAmountUSD"]) for r in ROWS)
    assert total == 350_000_000
    assert accepts("debt_04", "$350,000,000") and accepts("debt_04", "$350 million")
    assert not accepts("debt_04", "$3,599,000,000")  # total FINANCING, not debt


def test_debt_05_all_in_rate():
    spread_pct = int(by_name("Apex Software")["SpreadOverSOFRbps"]) / 100
    assert round(4.30 + spread_pct, 2) == 8.55
    assert accepts("debt_05", "4.30% + 4.25% = 8.55%")
    assert not accepts("debt_05", "4.30% + 425% = 429.30%")


def test_debt_06_price_is_really_missing():
    assert not any("price" in c.lower() for c in ROWS[0])
    assert accepts("debt_06", "The answer isn't in the data.")


def test_no_company_name_hides_inside_another():
    """'Zenith AI' must not match inside 'Zenith AI Labs' etc., or checks would lie."""
    all_names = {r["CompanyName"] for r in ROWS}
    for case_id in ("debt_02", "debt_03"):
        for name in required(case_id) | forbidden(case_id):
            assert not any(name != other and name.lower() in other.lower() for other in all_names), name
