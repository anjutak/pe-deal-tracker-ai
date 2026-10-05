"""
Step 5 - Flakiness check: ask every golden question several times and report pass RATES.

An LLM can answer the same question differently each time, so one pass/fail
result isn't the whole story. This script shows which questions are:
  STABLE PASS  - passed every run
  FLAKY        - passed some runs, failed others   <- the interesting ones
  STABLE FAIL  - failed every run (a consistent bug)

Usage (same environment variables as the pytest suite):
    python run_repeats.py                      # all 27 questions x 3 runs, keyword checks
    python run_repeats.py --runs 5             # more runs = more reliable rates
    python run_repeats.py --only math ambig    # only cases whose id starts with these
    python run_repeats.py --judge              # also score every answer with the Claude judge

Results: printed table + results/flakiness.json (every answer from every run).
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from test_deal_tracker import CASES, REQUEST_DELAY, ask_deal_tracker, contains_any


def keyword_pass(case: dict, answer: str) -> bool:
    has_all = all(contains_any(answer, group) for group in case["must_contain"])
    has_none = not any(contains_any(answer, group) for group in case["must_not_contain"])
    return has_all and has_none


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="times to ask each question (default 3)")
    parser.add_argument("--only", nargs="*", help="only case ids starting with these prefixes, e.g. math ambig")
    parser.add_argument("--judge", action="store_true", help="also score each answer with the Claude judge")
    args = parser.parse_args()

    if not os.getenv("LANGFLOW_URL") or not os.getenv("LANGFLOW_API_KEY"):
        raise SystemExit("Set LANGFLOW_URL and LANGFLOW_API_KEY first.")
    if args.judge:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("--judge needs ANTHROPIC_API_KEY.")
        from judge import judge_answer

    cases = [c for c in CASES if not args.only or c["id"].startswith(tuple(args.only))]
    total_calls = len(cases) * args.runs
    print(f"Asking {len(cases)} questions x {args.runs} runs = {total_calls} flow calls "
          f"(about {total_calls * (REQUEST_DELAY + 5) / 60:.0f} minutes)\n")

    results = []
    consecutive_errors = 0
    for case in cases:
        runs = []
        for i in range(1, args.runs + 1):
            time.sleep(REQUEST_DELAY)
            run = {"run": i}
            try:
                run["answer"] = ask_deal_tracker(case["question"])
                run["keyword_pass"] = keyword_pass(case, run["answer"])
                if args.judge:
                    v = judge_answer(case, run["answer"])
                    run["judge_pass"] = v["passed"]
                    run["judge_scores"] = f"C{v['correctness']} G{v['grounding']} S{v['scope']}"
                    run["judge_issues"] = v["issues"]
                consecutive_errors = 0
            except Exception as e:  # infrastructure problem - recorded, not counted as a fail
                run["error"] = f"{type(e).__name__}: {e}"[:500]
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"\n  ERROR on {case['id']}: {run['error']}\n  ", end="")
                stop_reason = None
                if "RESOURCE_EXHAUSTED" in str(e):
                    stop_reason = "Gemini quota is used up"
                elif consecutive_errors >= 3:
                    stop_reason = "3 errors in a row - the flow isn't reachable or is failing"
                if stop_reason:
                    print(f"\n\n{stop_reason}. Stopping early (partial results are saved).")
                    print(f"Last error: {run['error']}")
                    results.append({"case": case, "runs": runs + [run]})
                    return save_and_report(results, args)
            runs.append(run)
            mark = "E" if "error" in run else ("." if run["keyword_pass"] else "F")
            print(mark, end="", flush=True)
        print(f"  {case['id']}")
        results.append({"case": case, "runs": runs})

    save_and_report(results, args)


def rate(runs: list, key: str):
    scored = [r for r in runs if key in r]
    return (sum(r[key] for r in scored), len(scored)) if scored else (0, 0)


def status(passed: int, total: int) -> str:
    if total == 0:
        return "NO DATA"
    if passed == total:
        return "STABLE PASS"
    if passed == 0:
        return "STABLE FAIL"
    return "FLAKY"


def save_and_report(results: list, args):
    summary = []
    print(f"\n{'Case':<11} {'Keyword':<9} {'Status':<12}" + (f" {'Judge':<7} {'Status':<12}" if args.judge else ""))
    print("-" * (34 + (21 if args.judge else 0)))
    for item in results:
        case, runs = item["case"], item["runs"]
        kp, kt = rate(runs, "keyword_pass")
        row = {"id": case["id"], "category": case["category"],
               "keyword_passed": kp, "keyword_runs": kt, "keyword_status": status(kp, kt),
               "errors": sum("error" in r for r in runs)}
        line = f"{case['id']:<11} {kp}/{kt:<7} {row['keyword_status']:<12}"
        if args.judge:
            jp, jt = rate(runs, "judge_pass")
            row.update(judge_passed=jp, judge_runs=jt, judge_status=status(jp, jt))
            line += f" {jp}/{jt:<5} {row['judge_status']:<12}"
        # Distinct answers show HOW it varies (e.g. a different wrong total each run)
        row["distinct_answers"] = len({r["answer"].strip() for r in runs if "answer" in r})
        summary.append(row)
        print(line)

    counts = Counter(r["keyword_status"] for r in summary)
    print(f"\nKeyword checks: {counts['STABLE PASS']} stable pass, {counts['FLAKY']} flaky, "
          f"{counts['STABLE FAIL']} stable fail")
    if args.judge:
        jc = Counter(r["judge_status"] for r in summary)
        print(f"Judge:          {jc['STABLE PASS']} stable pass, {jc['FLAKY']} flaky, {jc['STABLE FAIL']} stable fail")
    errors = sum(r["errors"] for r in summary)
    if errors:
        print(f"Infrastructure errors (not counted as fails): {errors}")

    out = Path(__file__).parent / "results" / "flakiness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "runs_per_case": args.runs, "summary": summary,
        "details": [{"id": i["case"]["id"], "question": i["case"]["question"],
                     "expected_answer": i["case"]["expected_answer"], "runs": i["runs"]} for i in results],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
