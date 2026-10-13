"""
Score saved answers with DeepEval, and compare it with our own Claude judge.

Uses answers you ALREADY have (a results/judge_results.json from pytest), so
it makes no new Langflow calls. Each answer gets three DeepEval metrics:

  Faithfulness     - are the answer's claims supported by the deal data?
                     (Given only the rows for companies named in the question or
                     answer: DeepEval first lists every fact in its context, and the
                     whole 50 x 16 table is ~800 facts, too many for one reply.
                     Skipped when no company is named, e.g. totals and refusals.)
  AnswerRelevancy  - does the answer address the question?
  Correctness      - GEval with our own criteria: does it match the expected
                     answer (all names, numbers and counts; nothing extra and wrong)?

Then it shows where DeepEval and our judge agree and disagree. The
disagreements are the interesting part: they show what each tool can and
can't catch.

Setup (one time):   pip install -r requirements-deepeval.txt
Run (try 3 cases):  python deepeval_check.py results/full_run_v2_debt_data.json --only math_08 debt_02 debt_05
Run everything:     python deepeval_check.py results/full_run_v2_debt_data.json
Redo only errors:   python deepeval_check.py results/full_run_v2_debt_data.json --resume

Needs ANTHROPIC_API_KEY. DEALS_CSV picks the data file (default: deals_with_debt.csv).
DEEPEVAL_MODEL picks the Claude model (default: claude-sonnet-5).
Results: printed table + results/deepeval_results.json
"""
import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval  # noqa: E402
from deepeval.models import AnthropicModel  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from compare_runs import keyword_pass  # noqa: E402  (same keyword rule as the suite)

HERE = Path(__file__).parent
CASES = {c["id"]: c for c in json.loads((HERE / "golden_dataset.json").read_text(encoding="utf-8"))["cases"]}
DEALS_CSV = Path(os.getenv("DEALS_CSV", HERE / "deals_with_debt.csv"))
MODEL = os.getenv("DEEPEVAL_MODEL", "claude-sonnet-5")
THRESHOLD = 0.7
MAX_FAITHFULNESS_ROWS = 12  # more rows = too many facts for DeepEval to list in one reply

# Our own definition of "correct" for this app, so DeepEval checks what matters here.
CORRECTNESS_STEPS = [
    "Compare the actual output with the expected output for the given input.",
    "Every company name, number, count and total in the expected output must appear in the actual output with the same value.",
    "Heavily penalize missing items in a list, extra items that should not be there, and wrong numbers or totals.",
    "Heavily penalize internal contradictions, e.g. a stated count that differs from the number of items listed.",
    "If the expected output says the answer is not in the data or the request should be declined, the actual output must decline and must not invent an answer.",
    "Extra correct detail and different wording or formatting are fine.",
]


def build_metrics(model):
    return {
        "faithfulness": FaithfulnessMetric(threshold=THRESHOLD, model=model, async_mode=False),
        "relevancy": AnswerRelevancyMetric(threshold=THRESHOLD, model=model, async_mode=False),
        "correctness": GEval(
            name="Correctness", evaluation_steps=CORRECTNESS_STEPS, model=model,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            threshold=THRESHOLD, async_mode=False,
        ),
    }


def relevant_rows(deal_data: str, *texts: str) -> str:
    """The CSV header plus the rows of every company named in the texts ("" if none)."""
    reader = csv.reader(io.StringIO(deal_data))
    header, rows = next(reader), list(reader)
    joined = " ".join(texts).lower()
    hits = [r for r in rows if r[0].lower() in joined]
    if not hits or len(hits) > MAX_FAITHFULNESS_ROWS:
        return ""
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([header] + hits)
    return buf.getvalue()


def score(metrics: dict, test_case: LLMTestCase) -> dict:
    out = {}
    for name, metric in metrics.items():
        if name == "faithfulness" and not test_case.retrieval_context:
            out[name] = {"score": None, "passed": None, "skipped": True,
                         "reason": f"No company named, or more than {MAX_FAITHFULNESS_ROWS} (too many facts for one reply)."}
            continue
        try:
            metric.measure(test_case)
            out[name] = {"score": round(metric.score, 2), "passed": metric.is_successful(), "reason": metric.reason}
        except Exception as e:  # one broken metric shouldn't lose the whole run
            out[name] = {"score": None, "passed": None, "reason": f"ERROR {type(e).__name__}: {e}"[:300]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="a judge_results.json saved by pytest")
    parser.add_argument("--only", nargs="*", help="only these case ids (start small: each case is several API calls)")
    parser.add_argument("--resume", action="store_true",
                        help="keep finished cases from the last results/deepeval_results.json; redo only cases with errors")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    saved = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if "results" not in saved:
        sys.exit(f"{args.results} is not a judge_results.json from pytest.")
    rows = [r for r in saved["results"] if r["id"] in CASES and (not args.only or r["id"] in args.only)]

    out = HERE / "results" / "deepeval_results.json"
    done = {}
    if args.resume and out.exists():
        done = {x["id"]: x for x in json.loads(out.read_text(encoding="utf-8"))["results"]
                if not any(m["score"] is None and not m.get("skipped") for m in x["deepeval"].values())}
        print(f"Resuming: keeping {len(done)} finished cases, redoing {len([r for r in rows if r['id'] not in done])}.")

    deal_data = DEALS_CSV.read_text(encoding="utf-8")
    model = AnthropicModel(model=MODEL, api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"DeepEval ({MODEL}) on {len(rows)} saved answers, data: {DEALS_CSV.name}\n")
    print(f"{'Case':<11} {'Keyword':<8} {'Our judge':<10} {'Faithful':<9} {'Relevant':<9} {'Correct':<8} DeepEval")
    print("-" * 72)

    results = []
    cell = lambda m: "n/a" if m.get("skipped") else "ERR" if m["score"] is None else f"{m['score']:.2f}{'' if m['passed'] else ' x'}"  # noqa: E731
    first_error = None
    for r in rows:
        if r["id"] in done:
            row, de = done[r["id"]], done[r["id"]]["deepeval"]
            results.append(row)
            print(f"{r['id']:<11} {'pass' if row['keyword_pass'] else 'FAIL':<8} {'pass' if row['our_judge_pass'] else 'FAIL':<10} "
                  f"{cell(de['faithfulness']):<9} {cell(de['relevancy']):<9} {cell(de['correctness']):<8} "
                  f"{ {True: 'PASS', False: 'FAIL', None: 'ERROR'}[row['deepeval_pass']]}  (kept)")
            continue
        case = CASES[r["id"]]
        rows_ctx = relevant_rows(deal_data, case["question"], r["answer"])
        test_case = LLMTestCase(input=case["question"], actual_output=r["answer"],
                                expected_output=case["expected_answer"],
                                retrieval_context=[rows_ctx] if rows_ctx else None)
        # Fresh metric objects for every case: a metric that errored once keeps
        # reporting "failed" afterwards, even for perfect scores.
        de = score(build_metrics(model), test_case)
        used = [m for m in de.values() if not m.get("skipped")]
        de_pass = None if any(m["passed"] is None for m in used) else all(m["passed"] for m in used)
        row = {"id": r["id"], "question": case["question"], "answer": r["answer"],
               "keyword_pass": keyword_pass(r["id"], r["answer"]), "our_judge_pass": r["passed"],
               "our_judge_issues": r.get("issues", []), "deepeval_pass": de_pass, "deepeval": de}
        results.append(row)
        if first_error is None:
            first_error = next((m["reason"] for m in de.values() if m["score"] is None and not m.get("skipped")), None)
        verdict = {True: "PASS", False: "FAIL", None: "ERROR"}[de_pass]
        print(f"{r['id']:<11} {'pass' if row['keyword_pass'] else 'FAIL':<8} {'pass' if r['passed'] else 'FAIL':<10} "
              f"{cell(de['faithfulness']):<9} {cell(de['relevancy']):<9} {cell(de['correctness']):<8} {verdict}", flush=True)

    scored = [x for x in results if x["deepeval_pass"] is not None]
    agree = [x for x in scored if x["deepeval_pass"] == x["our_judge_pass"]]
    only_ours = [x["id"] for x in scored if not x["our_judge_pass"] and x["deepeval_pass"]]
    only_de = [x["id"] for x in scored if x["our_judge_pass"] and not x["deepeval_pass"]]
    print(f"\nAgreement with our judge: {len(agree)}/{len(scored)}")
    print(f"Only OUR JUDGE failed it:  {', '.join(only_ours) or 'none'}")
    print(f"Only DEEPEVAL failed it:   {', '.join(only_de) or 'none'}")
    for name in ("faithfulness", "relevancy", "correctness"):
        caught = [x["id"] for x in scored if not x["our_judge_pass"] and x["deepeval"][name]["passed"] is False]
        failed_ours = sum(not x["our_judge_pass"] for x in scored)
        print(f"  {name:<13} caught {len(caught)}/{failed_ours} of the answers our judge failed")
    print("\n(x = below the 0.7 threshold. n/a = faithfulness skipped: no company named, or too many.) Read the reasons for each disagreement in the JSON file before deciding which tool was right.")

    if first_error:
        print(f"\nFirst error: {first_error}")
        print("Fix the cause (e.g. wait for a rate limit, add API credits), then re-run with --resume.")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run_at": datetime.now().isoformat(timespec="seconds"), "model": MODEL,
                               "source": str(args.results), "threshold": THRESHOLD, "results": results},
                              indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
