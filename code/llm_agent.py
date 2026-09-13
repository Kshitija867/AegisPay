"""Gemini-backed, closed-world plan selection for AegisPay.

Financial calculations and candidate construction stay in deterministic Python.
This module only chooses among already validated candidates and produces a
concise, structured explanation for a separately verified selected candidate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

try:
    from google import genai
    from google.genai import types
except ImportError:  # Allows imports and deterministic tests before dependency installation.
    genai = None
    types = None


DEFAULT_MODEL = "gemini-3.8-flash"
SYSTEM_GUARDRAIL = (
    "You are a financial planning decision agent. You are NOT a calculator. "
    "All numerical financial facts supplied in the context are authoritative. "
    "Choose only one candidate_plan_id from the supplied candidates. Never invent "
    "a candidate, amount, date, payment, spending change, or financial fact."
)


class LLMConfigurationError(RuntimeError):
    """Raised when Gemini configuration is incomplete or the SDK is unavailable."""


class LLMRequestError(RuntimeError):
    """Raised for a Gemini request failure without exposing credentials."""


class LLMDecision(BaseModel):
    """Only fields that deterministic code can validate before use."""

    selected_candidate_id: str = Field(min_length=1)
    reasoning: str = Field(min_length=1, max_length=600)
    confidence: Literal["high", "medium", "low"]


class _LLMExplanation(BaseModel):
    explanation: str = Field(min_length=1, max_length=600)


@dataclass(frozen=True)
class TokenUsage:
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    api_calls: int


def _usage_value(usage: Any, *names: str) -> int | None:
    """Read SDK usage fields defensively; absent metadata remains unknown."""
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return None


def _json_context(value: Any) -> str:
    """Stable serialization for prompt context; dates/Decimals become strings."""
    return json.dumps(value, default=str, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _candidate_ids(candidate_plans: list[dict[str, Any]]) -> set[str]:
    if not candidate_plans:
        raise ValueError("candidate_plans must contain a deterministic fallback candidate")
    candidate_ids: set[str] = set()
    for candidate in candidate_plans:
        candidate_id = candidate.get("candidate_plan_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("each candidate plan requires a non-empty candidate_plan_id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_plan_id: {candidate_id!r}")
        candidate_ids.add(candidate_id)
    return candidate_ids


class GeminiPlanSelector:
    """Small synchronous Gemini wrapper with bounded retries and usage metadata."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 2,
    ) -> None:
        load_dotenv()
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        # This integration is intentionally configured only through GEMINI_API_KEY.
        # Remove a process-local alternate key so the SDK cannot silently prefer it.
        os.environ.pop("GOOGLE_API_KEY", None)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._client: Any | None = None
        self.last_usage = TokenUsage(self.model, None, None, None, 0)

    def _client_or_raise(self) -> Any:
        if genai is None or types is None:
            raise LLMConfigurationError("Gemini SDK is unavailable; install dependencies from requirements.txt")
        if not self._api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not configured")
        if self._client is None:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(client_args={"timeout": self.timeout_seconds}),
            )
        return self._client

    def choose_plan(
        self,
        request_context: dict[str, Any],
        financial_facts: dict[str, Any],
        candidate_plans: list[dict[str, Any]],
    ) -> LLMDecision:
        valid_ids = _candidate_ids(candidate_plans)
        prompt = self._selection_prompt(request_context, financial_facts, candidate_plans)
        decision = self._structured_request(prompt, LLMDecision)
        if decision.selected_candidate_id not in valid_ids:
            raise LLMRequestError("Gemini selected an ID outside the supplied candidate plans")
        return decision

    def explain_decision(
        self,
        request_context: dict[str, Any],
        financial_facts: dict[str, Any],
        selected_candidate: dict[str, Any],
    ) -> str:
        candidate_id = _candidate_ids([selected_candidate]).pop()
        prompt = (
            f"{SYSTEM_GUARDRAIL}\n"
            "Write one concise explanation (60 words or fewer) for the supplied selected "
            "candidate. Mention only facts supplied below. Do not introduce a number, date, "
            "payment, spending change, or recommendation not present in the selected candidate "
            "or financial facts.\n"
            f"REQUEST_CONTEXT={_json_context(request_context)}\n"
            f"FINANCIAL_FACTS={_json_context(financial_facts)}\n"
            f"SELECTED_CANDIDATE_ID={candidate_id}\n"
            f"SELECTED_CANDIDATE={_json_context(selected_candidate)}"
        )
        explanation = self._structured_request(prompt, _LLMExplanation).explanation.strip()
        if len(explanation.split()) > 60:
            raise LLMRequestError("Gemini explanation exceeded the 60-word limit")
        return explanation

    def _selection_prompt(
        self,
        request_context: dict[str, Any],
        financial_facts: dict[str, Any],
        candidate_plans: list[dict[str, Any]],
    ) -> str:
        return (
            f"{SYSTEM_GUARDRAIL}\n"
            "The candidate plans are a closed world. Return one supplied candidate_plan_id only.\n"
            f"REQUEST_CONTEXT={_json_context(request_context)}\n"
            f"FINANCIAL_FACTS={_json_context(financial_facts)}\n"
            f"CANDIDATE_PLANS={_json_context(candidate_plans)}"
        )

    def _structured_request(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        client = self._client_or_raise()
        response: Any | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=256,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                break
            except Exception as exc:
                if attempt == self.max_attempts:
                    # Do not include SDK text: it may contain request credentials or URLs.
                    status = getattr(exc, "status_code", None)
                    status_suffix = f", status={status}" if isinstance(status, int) else ""
                    raise LLMRequestError(
                        f"Gemini request failed after {self.max_attempts} attempt(s): "
                        f"{type(exc).__name__}{status_suffix}"
                    ) from exc
                time.sleep(0.25 * attempt)
        assert response is not None
        self._record_usage(response)
        try:
            parsed = getattr(response, "parsed", None)
            return parsed if isinstance(parsed, schema) else schema.model_validate_json(response.text)
        except (ValidationError, TypeError, ValueError, AttributeError) as exc:
            raise LLMRequestError("Gemini returned invalid structured output") from exc

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        input_tokens = _usage_value(usage, "prompt_token_count", "input_token_count")
        output_tokens = _usage_value(usage, "candidates_token_count", "output_token_count")
        total_tokens = _usage_value(usage, "total_token_count")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        self.last_usage = TokenUsage(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            api_calls=self.last_usage.api_calls + 1,
        )


def choose_plan(
    request_context: dict[str, Any],
    financial_facts: dict[str, Any],
    candidate_plans: list[dict[str, Any]],
) -> LLMDecision:
    """Choose one pre-computed candidate through a freshly configured selector."""
    return GeminiPlanSelector().choose_plan(request_context, financial_facts, candidate_plans)


def explain_decision(
    request_context: dict[str, Any], financial_facts: dict[str, Any], selected_candidate: dict[str, Any]
) -> str:
    """Generate a concise explanation for an already selected deterministic candidate."""
    return GeminiPlanSelector().explain_decision(request_context, financial_facts, selected_candidate)


def _smoke_test() -> None:
    """Run exactly one safe synthetic call when credentials are configured."""
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        print("Gemini smoke test skipped: GEMINI_API_KEY is not configured.")
        return
    selector = GeminiPlanSelector(max_attempts=1)
    try:
        decision = selector.choose_plan(
            request_context={"request_type": "purchase", "user_goal": "Choose a supplied safe option."},
            financial_facts={"deterministic_status": "Both candidates are verified by Python."},
            candidate_plans=[
                {"candidate_plan_id": "candidate_wait", "summary": "Wait for the verified later option."},
                {"candidate_plan_id": "candidate_full", "summary": "Use the verified full-payment option."},
            ],
        )
    except (LLMConfigurationError, LLMRequestError) as exc:
        print(f"Gemini smoke test unavailable: {exc}")
        return
    print(
        "Gemini smoke test passed: "
        f"selected={decision.selected_candidate_id}, model={selector.last_usage.model}, "
        f"input_tokens={selector.last_usage.input_tokens}, output_tokens={selector.last_usage.output_tokens}, "
        f"total_tokens={selector.last_usage.total_tokens}, api_calls={selector.last_usage.api_calls}"
    )


if __name__ == "__main__":
    _smoke_test()
