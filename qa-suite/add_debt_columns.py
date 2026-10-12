"""
Step 1: add venture-debt columns to the deal data.

Reads the original deal CSV and writes a NEW file with 5 extra columns.
The original file is not changed, so the existing 27 tests keep working.

New columns:
  DebtType           - "Venture Loan", "Convertible Note", or "None"
  DebtAmountUSD      - amount borrowed (0 if no debt)
  RateType           - "Fixed" or "Floating" (blank if no debt)
  FixedRatePct       - interest rate in percent, e.g. 9.5 means 9.5% (fixed-rate debt only)
  SpreadOverSOFRbps  - extra interest over SOFR in basis points, e.g. 475 means +4.75%
                       (floating-rate debt only; 100 basis points = 1%)
  MaturityYear       - year the debt must be paid back

Run:  python add_debt_columns.py
The random seed is fixed, so every run produces the same file.
"""
import csv
import random
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "fake_investment_pipeline_unique_companies.csv"
OUTPUT = HERE / "deals_with_debt.csv"

NEW_COLUMNS = ["DebtType", "DebtAmountUSD", "RateType",
               "FixedRatePct", "SpreadOverSOFRbps", "MaturityYear"]

rng = random.Random(42)  # fixed seed = same data every time

with SOURCE.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames + NEW_COLUMNS
    rows = list(reader)

for row in rows:
    # About half the companies have no debt.
    debt_type = rng.choice(["None", "None", "Venture Loan", "Convertible Note"])
    row["DebtType"] = debt_type

    if debt_type == "None":
        row.update(DebtAmountUSD=0, RateType="", FixedRatePct="",
                   SpreadOverSOFRbps="", MaturityYear="")
        continue

    # Debt is 10-30% of the round size, rounded to the nearest $1M.
    round_size = int(row["FinancingAmountUSD"])
    row["DebtAmountUSD"] = round(round_size * rng.uniform(0.10, 0.30), -6)
    row["DebtAmountUSD"] = int(row["DebtAmountUSD"])
    row["MaturityYear"] = rng.randint(2027, 2033)

    if debt_type == "Convertible Note":
        # Convertible notes usually carry a fixed rate.
        row["RateType"] = "Fixed"
        row["FixedRatePct"] = rng.choice([6.0, 7.5, 8.0, 9.5, 10.0])
        row["SpreadOverSOFRbps"] = ""
    else:
        # Venture loans here are floating: SOFR + a spread.
        row["RateType"] = "Floating"
        row["FixedRatePct"] = ""
        row["SpreadOverSOFRbps"] = rng.choice([350, 425, 475, 550, 625])

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

with_debt = [r for r in rows if r["DebtType"] != "None"]
print(f"Wrote {OUTPUT.name}: {len(rows)} deals, {len(with_debt)} with debt "
      f"({sum(r['RateType'] == 'Fixed' for r in with_debt)} fixed, "
      f"{sum(r['RateType'] == 'Floating' for r in with_debt)} floating)")
