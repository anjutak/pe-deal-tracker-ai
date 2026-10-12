"""
Compare two test runs and show what changed - so nobody has to read result
files by hand to spot a regression.

Works with both result files the suite writes:
  results/judge_results.json  (from pytest: one answer per case)
  results/flakiness.json      (from run_repeats.py: several answers per case)
For pytest files the keyword check is recomputed from the saved answer using
golden_dataset.json, so both checks can be compared.

Usage:
    python compare_runs.py BEFORE.json AFTER.json
    python compare_runs.py BEFORE.json AFTER.json --fail-on-regression   # exit 1 if anything broke (for CI)

A case is "newly broken" if it passed every time before and doesn't now,
on the keyword check, the judge, or both.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CASES = {c["id"]: c for c in json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))["cases"]}

# Order matters: a case is put in the first group that applies.
GROUPS = [
    ("broken", "NEWLY BROKEN  (passed every time before, fails now)"),
    ("worse", "GOT WORSE     (was already flaky, now fails more often)"),
    ("fixed", "NEWLY FIXED   (failed before, passes every time now)"),
    ("better", "GOT BETTER    (passes more often, but not every time)"),
    ("still_failing", "STILL FAILING (failed every time in both runs: known bugs)"),
    ("same", "NO CHANGE"),
]


def keyword_pass(case_id: str, answer: str) -> bool:
    """Same rule as test_golden_case: every must_contain group found, no must_not_contain group found."""
    case, text = CASES[case_id], answer.lower()
    found = lambda group: any(alt.lower() in text for alt in group)  # noqa: E731
    return all(found(g) for g in case["must_contain"]) and not any(found(g) for g in case["must_not_contain"])


def load(path: str) -> dict:
    """Read either result format into {case_id: {"keyword": [bool, ...], "judge": [bool, ...]}}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    if "details" in data:  # run_repeats.py format
        for item in data["details"]:
            runs = [r for r in item["runs"] if "error" not in r]  # errors are infrastructure, not fails
            out[item["id"]] = {"keyword": [r["keyword_pass"] for r in runs],
                               "judge": [r["judge_pass"] for r in runs if "judge_pass" in r]}
    elif "results" in data:  # pytest judge_results.json format
        for r in data["results"]:
            kw = [keyword_pass(r["id"], r["answer"])] if r["id"] in CASES else []
            out[r["id"]] = {"keyword": kw, "judge": [r["passed"]]}
    else:
        sys.exit(f"{path}: not a results file from pytest or run_repeats.py")
    return out


def fmt(results: list) -> str:
    return f"{sum(results)}/{len(results)}" if results else "-"


def change(before: list, after: list):
    """How one check (keyword or judge) changed. None if either run has no data for it."""
    if not before or not after:
        return None
    b, a = sum(before) / len(before), sum(after) / len(after)
    if b == 1 and a < 1:
        return "broken"
    if b < 1 and a == 1:
        return "fixed"
    if b == 0 and a == 0:
        return "still_failing"
    if a < b:
        return "worse"
    if a > b:
        return "better"
    return "same"


def classify(before: dict, after: dict) -> str:
    """Put the case in the most important group either check falls into."""
    changes = {change(before[k], after[k]) for k in ("keyword", "judge")} - {None}
    for group, _ in GROUPS:
        if group in changes:
            return group
    return "same"


def compare(before: dict, after: dict) -> dict:
    groups = {g: [] for g, _ in GROUPS}
    for case_id in sorted(before.keys() & after.keys()):
        groups[classify(before[case_id], after[case_id])].append(case_id)
    groups["new"] = sorted(after.keys() - before.keys())
    groups["missing"] = sorted(before.keys() - after.keys())
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--fail-on-regression", action="store_true", help="exit with code 1 if any case broke or got worse")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    groups = compare(before, after)

    print(f"Before: {args.before}\nAfter:  {args.after}")
    for group, title in GROUPS:
        ids = groups[group]
        print(f"\n{title}: {len(ids)}")
        if group == "same":
            continue  # list only the changes
        for case_id in ids:
            b, a = before[case_id], after[case_id]
            print(f"  {case_id:<11} keyword {fmt(b['keyword']):>4} -> {fmt(a['keyword']):<4}"
                  f"  judge {fmt(b['judge']):>4} -> {fmt(a['judge']):<4}"
                  f"  {CASES.get(case_id, {}).get('question', '')[:55]}")

    if groups["new"]:
        print(f"\nNEW CASES (only in the after run): {len(groups['new'])}")
        for case_id in groups["new"]:
            a = after[case_id]
            print(f"  {case_id:<11} keyword {fmt(a['keyword']):<4}  judge {fmt(a['judge']):<4}")
    if groups["missing"]:
        print(f"\nNOT IN THE AFTER RUN: {', '.join(groups['missing'])}")

    regressions = groups["broken"] + groups["worse"]
    one_run = [c for c in regressions if len(after[c]["judge"] or after[c]["keyword"]) == 1]
    if one_run:
        # One answer can be bad luck. Re-asking only these cases is quick and settles it.
        print("\nThese regressions come from a single answer. Confirm them with:")
        print(f"  python run_repeats.py --only {' '.join(one_run)} --runs 5 --judge")

    if args.fail_on_regression and regressions:
        print(f"\nFAIL: {len(regressions)} regression(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
