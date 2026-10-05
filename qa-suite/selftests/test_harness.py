"""
Tests for the test harness itself, against a fake Langflow server and a fake
Claude client. Checks the behaviors the live suite depends on: answer parsing,
retries, quota handling, keyword matching, judge verdict parsing, and kappa.

Runs offline - no Langflow, no API keys.
"""
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import anthropic
import pytest
import requests

import compare_judge
import judge
import test_deal_tracker as suite


# ---- Fake Langflow server -------------------------------------------------------
class FakeLangflow:
    """Replies with queued (status, body) responses; records every request."""

    def __init__(self):
        self.queue, self.requests = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"body": body, "api_key": self.headers.get("x-api-key")})
                status, payload = fake.queue.pop(0) if fake.queue else (200, answer_payload("default"))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode() if isinstance(payload, dict) else payload.encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/v1/run/flow"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def answer_payload(text):
    return {"outputs": [{"outputs": [{"results": {"message": {"text": text}}}]}]}


@pytest.fixture
def langflow(monkeypatch):
    fake = FakeLangflow()
    monkeypatch.setattr(suite, "LANGFLOW_URL", fake.url)
    monkeypatch.setattr(suite, "LANGFLOW_API_KEY", "test-key")
    monkeypatch.setattr(suite.time, "sleep", lambda s: None)  # no real waiting between retries
    yield fake
    fake.server.shutdown()


# ---- Flow client -----------------------------------------------------------------
def test_returns_answer_text_and_sends_key_and_fresh_session(langflow):
    langflow.queue += [(200, answer_payload("Patel")), (200, answer_payload("Kim"))]
    assert suite.ask_deal_tracker("Q1") == "Patel"
    assert suite.ask_deal_tracker("Q2") == "Kim"
    first, second = langflow.requests
    assert first["api_key"] == "test-key" and first["body"]["input_value"] == "Q1"
    assert first["body"]["session_id"] != second["body"]["session_id"]  # no memory leaks between questions


def test_retries_server_errors_then_succeeds(langflow):
    langflow.queue += [(500, "boom"), (500, "boom"), (200, answer_payload("ok"))]
    assert suite.ask_deal_tracker("Q") == "ok"
    assert len(langflow.requests) == 3


def test_gives_up_after_max_attempts_with_server_message(langflow):
    langflow.queue += [(500, "flow crashed")] * 3
    with pytest.raises(RuntimeError, match="flow crashed"):
        suite.ask_deal_tracker("Q")
    assert len(langflow.requests) == suite.MAX_ATTEMPTS


def test_quota_exhaustion_stops_immediately(langflow):
    # Langflow reports Gemini's 429 as a 500 - the text is what identifies it
    langflow.queue += [(500, '{"detail": "429 RESOURCE_EXHAUSTED quota"}')]
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        suite.ask_deal_tracker("Q")
    assert len(langflow.requests) == 1


def test_bad_api_key_is_not_retried(langflow):
    langflow.queue += [(403, "forbidden")]
    with pytest.raises(requests.HTTPError):
        suite.ask_deal_tracker("Q")
    assert len(langflow.requests) == 1


# ---- Keyword matching -------------------------------------------------------------
@pytest.mark.parametrize("answer, alternatives, expected", [
    ("The owner is **PATEL**.", ["Patel"], True),
    ("Total: $392,000,000", ["392M", "392,000,000"], True),
    ("Total: $391,000,000", ["392M", "392,000,000"], False),
])
def test_contains_any(answer, alternatives, expected):
    assert suite.contains_any(answer, alternatives) is expected


# ---- Judge verdict handling (fake Claude client) ------------------------------------
def fake_client(monkeypatch, responder):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responder(kwargs)

    monkeypatch.setattr(judge, "_get_client", lambda: types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    return calls


def tool_reply(c, g, s):
    block = types.SimpleNamespace(type="tool_use", input={"reasoning": "r", "correctness": c, "grounding": g, "scope": s, "issues": []})
    return types.SimpleNamespace(content=[block], usage=types.SimpleNamespace(input_tokens=10, output_tokens=5))


CASE = {"id": "x", "category": "lookup", "question": "Who owns Pioneer Energy?", "expected_answer": "Patel", "notes": ""}


@pytest.mark.parametrize("scores, passed", [((5, 5, 5), True), ((4, 4, 4), True), ((5, 3, 5), False), ((1, 5, 5), False)])
def test_judge_pass_requires_every_score_at_least_4(monkeypatch, scores, passed):
    fake_client(monkeypatch, lambda kw: tool_reply(*scores))
    assert judge.judge_answer(CASE, "Patel")["passed"] is passed


def test_judge_sends_data_with_caching_and_answer_as_untrusted_text(monkeypatch):
    calls = fake_client(monkeypatch, lambda kw: tool_reply(5, 5, 5))
    judge.judge_answer(CASE, "Patel")
    request = calls[0]
    assert "Fusion Robotics" in request["system"][1]["text"]           # the CSV is included
    assert request["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "<chatbot_answer>\nPatel\n</chatbot_answer>" in request["messages"][0]["content"]


def test_judge_without_verdict_raises(monkeypatch):
    text_only = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="I think it's fine")],
                                      usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))
    fake_client(monkeypatch, lambda kw: text_only)
    with pytest.raises(RuntimeError, match="did not return a verdict"):
        judge.judge_answer(CASE, "Patel")


def test_judge_retries_without_temperature_when_model_rejects_it(monkeypatch):
    def responder(kw):
        if "temperature" in kw:
            raise anthropic.BadRequestError("temperature is not supported for this model",
                                            response=httpx.Response(400, request=httpx.Request("POST", "https://x")), body=None)
        return tool_reply(5, 5, 5)

    calls = fake_client(monkeypatch, responder)
    assert judge.judge_answer(CASE, "Patel")["passed"]
    assert "temperature" in calls[0] and "temperature" not in calls[1]


# ---- Agreement statistics -------------------------------------------------------------
def test_cohens_kappa_known_values():
    assert compare_judge.cohens_kappa([(True, True), (False, False)] * 5) == 1.0
    # The real labeling round: 9/12 agree, kappa 0.40
    pairs = [(True, True)] * 7 + [(False, False)] * 2 + [(True, False)] * 2 + [(False, True)]
    assert round(compare_judge.cohens_kappa(pairs), 2) == 0.40
