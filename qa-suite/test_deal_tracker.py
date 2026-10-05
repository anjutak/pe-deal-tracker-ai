"""
Deal Tracker QA suite.

Two tests run for every question in golden_dataset.json:
  test_golden_case - deterministic keyword checks (must_contain / must_not_contain)
  test_judge       - Claude scores correctness, grounding and scope (1-5 each)

Each question is sent to the flow only ONCE; both tests reuse the same answer.

Result meanings:
  PASSED - the answer met the checks
  FAILED - the flow answered, but the answer was wrong  (a quality finding)
  ERROR  - the flow or judge never answered (server error, quota, timeout)  (infrastructure)

Run everything:         pytest -v -rA
Keyword checks only:    pytest -v -rA -k golden
Judge only:             pytest -v -rA -k judge
Judge results are also saved to results/judge_results.json
"""
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import requests

# Load settings from a .env file next to this script, if there is one, so you
# don't have to type `set ...` in every new terminal. Real environment variables win.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from judge import judge_answer  # noqa: E402  (imported after .env so JUDGE_MODEL is picked up)

# ---- Settings (read from environment variables) ---------------------------
LANGFLOW_URL = os.getenv("LANGFLOW_URL")          # e.g. http://localhost:7860/api/v1/run/<flow-id>
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY")  # your Langflow API key (sk-...)

# Seconds to wait between questions. The Gemini free tier allows only a few
# requests per minute, so sending 27 questions back-to-back gets rate-limited.
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "7"))
MAX_ATTEMPTS = 3  # retries for server errors / timeouts

HERE = Path(__file__).parent
DATASET_PATH = HERE / "golden_dataset.json"
RESULTS_DIR = HERE / "results"
CASES = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["cases"]
CASE_IDS = [c["id"] for c in CASES]

_answer_cache: dict[str, str] = {}   # case id -> flow answer (so each question is asked once)
_judge_results: list[dict] = []      # collected for results/judge_results.json
_judge_broken: list[str] = []        # set if the judge request itself is invalid


# ---- API client ------------------------------------------------------------
def ask_deal_tracker(question: str) -> str:
    """Send one question to the Langflow flow and return the answer text.

    Retries on server errors (5xx), rate limits (429) and timeouts, waiting
    longer each time. Raises RuntimeError if every attempt fails.
    """
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                LANGFLOW_URL,
                headers={"x-api-key": LANGFLOW_API_KEY},
                json={
                    "input_value": question,
                    "input_type": "chat",
                    "output_type": "chat",
                    # New session per question so earlier answers can't leak into later ones
                    "session_id": str(uuid.uuid4()),
                },
                timeout=120,
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                # A used-up daily quota won't recover by retrying - stop right away
                if "RESOURCE_EXHAUSTED" in response.text:
                    break
            else:
                response.raise_for_status()  # other 4xx (e.g. 403 bad key) - don't retry
                data = response.json()
                return data["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except requests.exceptions.Timeout:
            last_error = "Timed out after 120 seconds"

        if attempt < MAX_ATTEMPTS:
            time.sleep(20 * attempt)  # 20s, then 40s

    raise RuntimeError(f"Flow API failed. Last error: {last_error}")


# ---- Helpers ---------------------------------------------------------------
def contains_any(answer: str, alternatives: list[str]) -> bool:
    """True if any alternative appears in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    return any(alt.lower() in answer_lower for alt in alternatives)


# ---- Fixtures --------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def check_settings_and_save_results():
    if not LANGFLOW_URL or not LANGFLOW_API_KEY:
        pytest.exit("Set LANGFLOW_URL and LANGFLOW_API_KEY environment variables first.", returncode=1)

    yield  # --- all tests run here ---

    if _judge_results:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "judge_results.json"
        out.write_text(json.dumps({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "passed": sum(r["passed"] for r in _judge_results),
            "total": len(_judge_results),
            "results": _judge_results,
        }, indent=2), encoding="utf-8")
        print(f"\nJudge results saved to {out}")


@pytest.fixture
def answer(case):
    """Get the flow's answer (cached). Runs as a fixture, so an API failure
    shows as ERROR (infrastructure) instead of FAILED (wrong answer)."""
    if case["id"] not in _answer_cache:
        time.sleep(REQUEST_DELAY)
        _answer_cache[case["id"]] = ask_deal_tracker(case["question"])
    return _answer_cache[case["id"]]


@pytest.fixture
def verdict(case, answer):
    """Get Claude's verdict. A judge API failure also shows as ERROR."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set - skipping the judge")
    if _judge_broken:
        # Same judge error on every case (e.g. bad request format) - don't repeat it 27 times
        pytest.skip(f"Judge failed on an earlier case: {_judge_broken[0]}")
    try:
        return judge_answer(case, answer)
    except Exception as e:
        if "BadRequest" in type(e).__name__ or "NotFound" in type(e).__name__:
            _judge_broken.append(str(e)[:200])
        raise


# ---- Tests -----------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_golden_case(case, answer):
    print(f"\nQ: {case['question']}\nA: {answer}\nExpected: {case['expected_answer']}")

    missing = [group for group in case["must_contain"] if not contains_any(answer, group)]
    forbidden = [group for group in case["must_not_contain"] if contains_any(answer, group)]

    assert not missing, f"Answer is missing expected content: {missing}"
    assert not forbidden, f"Answer contains forbidden content: {forbidden}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_judge(case, answer, verdict):
    _judge_results.append({
        "id": case["id"], "category": case["category"], "question": case["question"],
        "answer": answer, "expected_answer": case["expected_answer"], **verdict,
    })

    scores = f"correctness={verdict['correctness']} grounding={verdict['grounding']} scope={verdict['scope']}"
    print(f"\nQ: {case['question']}\nA: {answer}\n\nJudge: {scores}")
    print(f"Issues: {verdict['issues'] or 'none'}\nReasoning: {verdict['reasoning']}")

    assert verdict["passed"], f"Judge scores below 4: {scores}. Issues: {verdict['issues']}"
