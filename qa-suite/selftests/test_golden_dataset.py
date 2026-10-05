"""
Tests for the test data itself: recompute every expected answer in
golden_dataset.json directly from the CSV, so a typo in the golden dataset
can't silently turn a correct flow answer into a "failure" (or the reverse).

Runs offline - no Langflow, no API keys.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ROWS = list(csv.DictReader(open(ROOT / "fake_investment_pipeline_unique_companies.csv", encoding="utf-8")))
CASES = {c["id"]: c for c in json.loads((ROOT / "golden_dataset.json").read_text(encoding="utf-8"))["cases"]}
COMPANIES = {r["CompanyName"] for r in ROWS}


def companies_where(**conditions):
    return {r["CompanyName"] for r in ROWS if all(r[k] == v for k, v in conditions.items())}


def required_names(case_id):
    """Company names that a case's must_contain groups require."""
    return {g[0] for g in CASES[case_id]["must_contain"] if g[0] in COMPANIES}


def accepts(case_id, *texts):
    """True if each text satisfies at least one must_contain group's alternatives."""
    groups = CASES[case_id]["must_contain"]
    return all(any(alt.lower() in t.lower() for g in groups for alt in g) for t in texts)


# ---- Dataset structure -------------------------------------------------------
def test_ids_are_unique():
    ids = [c["id"] for c in json.loads((ROOT / "golden_dataset.json").read_text())["cases"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_case_is_well_formed(case_id):
    case = CASES[case_id]
    for field in ("id", "category", "question", "expected_answer", "must_contain", "must_not_contain"):
        assert field in case, f"{case_id} is missing {field}"
    for group in case["must_contain"] + case["must_not_contain"]:
        assert group and all(isinstance(alt, str) and alt for alt in group), f"{case_id} has an empty check group"


def test_csv_has_50_deals():
    assert len(ROWS) == 50


# ---- Lookups -------------------------------------------------------------------
def row(name):
    return next(r for r in ROWS if r["CompanyName"] == name)


def test_lookup_answers_match_csv():
    assert row("Pioneer Energy")["DealTeamMember"] == "Patel" and accepts("lookup_01", "Patel")
    assert row("Zenith AI")["Founders"] == "Mia, Mason"
    assert row("Helix Energy")["BusinessDescription"] == "Data analytics SaaS for enterprises"
    assert row("Echo Tech")["InvestmentStatus"] == "Due Diligence"
    assert row("Pulse Ventures")["FinancingAmountUSD"] == "145000000"
    assert row("Pulse Ventures")["FinancingRound"] == "Series C"
    assert row("Strata AI")["LatestValuationUSD"] == "2723292849"


# ---- Filters: required names must be exactly the right set ----------------------
@pytest.mark.parametrize("case_id, expected", [
    ("filter_01", companies_where(InvestmentStatus="Due Diligence")),
    ("filter_02", companies_where(InvestmentStatus="Investment Committee Voted")),
    ("filter_03", companies_where(DealTeamMember="Garcia")),
    ("filter_04", companies_where(FinancingRound="Series C", InvestmentStatus="Dead")),
    ("filter_05", {r["CompanyName"] for r in ROWS if "Liam" in r["Founders"].split(", ")}),
    ("ambig_01", {c for c in COMPANIES if c.startswith("Nova")}),
])
def test_filter_case_requires_exactly_the_right_companies(case_id, expected):
    assert required_names(case_id) == expected


def test_ambig_forbidden_names_are_not_nova_companies():
    forbidden = {g[0] for g in CASES["ambig_01"]["must_not_contain"]}
    assert forbidden <= COMPANIES and not any(n.startswith("Nova") for n in forbidden)


# ---- Aggregations: recompute the numbers -----------------------------------------
def amount(r):
    return int(r["FinancingAmountUSD"])


def test_math_01_total_deals():
    assert accepts("math_01", str(len(ROWS)))


def test_math_02_nda_count():
    assert str(len(companies_where(InvestmentStatus="NDA Signed"))) == "13" and accepts("math_02", "13")


def test_math_03_total_financing():
    total = sum(amount(r) for r in ROWS)
    assert total == 3_599_000_000 and accepts("math_03", f"{total:,}")


def test_math_04_closed_total():
    total = sum(amount(r) for r in ROWS if r["InvestmentStatus"] == "Closed")
    assert total == 392_000_000 and accepts("math_04", f"{total:,}", "$392M")


def test_math_05_and_06_largest_and_smallest():
    assert max(ROWS, key=amount)["CompanyName"] in required_names("math_05")
    assert min(ROWS, key=amount)["CompanyName"] in required_names("math_06")


def test_math_07_top_manager():
    (name, count), = Counter(r["DealTeamMember"] for r in ROWS).most_common(1)
    assert (name, count) == ("Kim", 10) and accepts("math_07", name, str(count))


def test_math_08_top_three_valuations():
    top3 = {r["CompanyName"] for r in sorted(ROWS, key=lambda r: -int(r["LatestValuationUSD"]))[:3]}
    assert required_names("math_08") == top3


# ---- Not-in-data cases must really be absent ------------------------------------
def test_nodata_cases_ask_for_missing_things():
    assert "Stellar Robotics" not in COMPANIES
    columns = {c.lower() for c in ROWS[0]}
    assert not any("close" in c and "date" in c for c in columns)
    assert not any("revenue" in c and "growth" not in c for c in columns)


def test_injection_case_uses_true_valuation():
    assert row("Pulse Ventures")["LatestValuationUSD"] == "2079241697"
    assert accepts("inject_02", "$2,079,241,697")
