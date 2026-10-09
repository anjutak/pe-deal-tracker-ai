"""
Step 3: add 6 debt test questions to golden_dataset.json.

Every expected answer is COMPUTED from deals_with_debt.csv with code,
never typed by hand, so the answers can't contain human typos.

Safe to run more than once: old debt_* cases are replaced, other cases
are left exactly as they are.

Run:  python add_debt_cases.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "deals_with_debt.csv"
DATASET = HERE / "golden_dataset.json"

ROWS = list(csv.DictReader(open(DATA, encoding="utf-8")))
WITH_DEBT = [r for r in ROWS if r["DebtType"] != "None"]
SOURCE_NOTE = "Computed from deals_with_debt.csv. "

# Words that mean "I don't have that information" (same idea as nodata_01).
NOT_IN_DATA = ["not in the data", "isn't in the data", "not available", "no information",
               "does not", "doesn't", "not included", "not provided"]


def row(name):
    return next(r for r in ROWS if r["CompanyName"] == name)


def names(rows):
    return [r["CompanyName"] for r in rows]


def money_variants(n):
    """Different ways a model might write the same dollar amount."""
    m = n / 1_000_000
    return [f"{n:,}", str(n), f"{m:,.0f} million", f"${m:,.0f}M", f"{m:,.0f}M"]


cases = []

# 1. Lookup: a floating rate written in basis points.
r = row("Pioneer Dynamics")
bps = int(r["SpreadOverSOFRbps"])
cases.append({
    "id": "debt_01", "category": "debt_lookup",
    "question": "What is the interest rate on Pioneer Dynamics' debt?",
    "expected_answer": f"Floating: SOFR + {bps} bps (SOFR + {bps / 100:.2f}%).",
    "must_contain": [["SOFR"], [str(bps), f"{bps / 100:.2f}"]],
    "must_not_contain": [],
    "notes": SOURCE_NOTE + "Checks the model reads the spread column and keeps it tied to SOFR.",
})

# 2. Filter with a boundary trap: "before 2030" must NOT include 2030.
before = [r for r in WITH_DEBT if int(r["MaturityYear"]) < 2030]
in_2030 = [r for r in WITH_DEBT if int(r["MaturityYear"]) == 2030]
cases.append({
    "id": "debt_02", "category": "debt_filter",
    "question": "Which deals have debt maturing before 2030?",
    "expected_answer": ", ".join(names(before)) + f" ({len(before)} deals).",
    "must_contain": [[n] for n in names(before)],
    "must_not_contain": [[n] for n in names(in_2030)],
    "notes": SOURCE_NOTE + "Boundary trap: deals maturing IN 2030 must be left out.",
})

# 3. Filter: fixed vs. floating.
fixed = [r for r in WITH_DEBT if r["RateType"] == "Fixed"]
floating = [r for r in WITH_DEBT if r["RateType"] == "Floating"]
cases.append({
    "id": "debt_03", "category": "debt_filter",
    "question": "Which deals have fixed-rate debt?",
    "expected_answer": ", ".join(names(fixed)) + f" ({len(fixed)} deals).",
    "must_contain": [[n] for n in names(fixed)],
    "must_not_contain": [[n] for n in names(floating)],
    "notes": SOURCE_NOTE + "No floating-rate deal may appear in the list.",
})

# 4. Aggregation: total debt (the flow already struggles with sums, see math_03).
total = sum(int(r["DebtAmountUSD"]) for r in ROWS)
cases.append({
    "id": "debt_04", "category": "debt_aggregation",
    "question": "What is the total debt amount across all deals?",
    "expected_answer": f"${total:,} across {len(WITH_DEBT)} deals with debt.",
    "must_contain": [money_variants(total)],
    "must_not_contain": [],
    "notes": SOURCE_NOTE + "Sum of DebtAmountUSD; deals with no debt count as 0.",
})

# 5. Calculation with a unit trap: basis points must be turned into percent.
r = row("Apex Software")
sofr = 4.30
bps = int(r["SpreadOverSOFRbps"])
all_in = sofr + bps / 100
wrong = sofr + bps  # the classic mistake: adding 425 as if it were 425%
cases.append({
    "id": "debt_05", "category": "debt_calculation",
    "question": "If SOFR is 4.30%, what is the all-in interest rate on Apex Software's loan?",
    "expected_answer": f"{sofr:.2f}% + {bps} bps ({bps / 100:.2f}%) = {all_in:.2f}%.",
    "must_contain": [[f"{all_in:.2f}"]],
    "must_not_contain": [[f"{wrong:.2f}"], [f"{wrong:.1f}%"]],
    "notes": SOURCE_NOTE + f"Unit trap: {bps} bps is {bps / 100:.2f}%, not {bps}%.",
})

# 6. Not in data: there is no market-price column.
assert not any("price" in c.lower() for c in ROWS[0]), "data now has a price column"
cases.append({
    "id": "debt_06", "category": "not_in_data",
    "question": "What is the current market price of Zenith Labs' loan?",
    "expected_answer": "Not in the data: the tracker has no market prices.",
    "must_contain": [NOT_IN_DATA],
    "must_not_contain": [],
    "notes": SOURCE_NOTE + "No price column exists; any number would be made up.",
})

# ---- Save: replace old debt_* cases, keep everything else ----------------------
dataset = json.loads(DATASET.read_text(encoding="utf-8"))
kept = [c for c in dataset["cases"] if not c["id"].startswith("debt_")]
dataset["cases"] = kept + cases
dataset["num_cases"] = len(dataset["cases"])
DATASET.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

print(f"golden_dataset.json now has {dataset['num_cases']} cases "
      f"({len(kept)} existing + {len(cases)} debt cases)\n")
for c in cases:
    print(f"{c['id']}: {c['question']}\n   -> {c['expected_answer']}\n")
