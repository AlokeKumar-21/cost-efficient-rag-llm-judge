
import json
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "judge_cases.json"
OUTPUT_PATH = BASE_DIR / "data" / "judge_results.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_judge")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)


JUDGE_INSTRUCTIONS = """
You are an impartial evaluator comparing two responses to a user question.

Evaluate each response on a 1-5 scale:
- correctness: factual accuracy and support
- relevance: how directly it answers the question
- instruction_following: compliance with explicit requirements
- clarity: understandable, appropriately concise communication

Use the question and the responses only. Do not prefer a response
because it is longer, more confident, or appears first.

Safety and honesty matter when relevant. Penalize fabricated facts
and unsafe recommendations.

Choose winner as A, B, or tie. Explain the decision briefly.
Scores must be integers from 1 to 5.

Return ONLY valid JSON with this structure:
{
  "scores": {
    "A": {
      "correctness": 1,
      "relevance": 1,
      "instruction_following": 1,
      "clarity": 1
    },
    "B": {
      "correctness": 1,
      "relevance": 1,
      "instruction_following": 1,
      "clarity": 1
    }
  },
  "winner": "A",
  "reason": "Brief explanation"
}
"""


def parse_json_response(text):
    """Parse JSON, tolerating Markdown code fences."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def judge_pair(question, response_a, response_b):
    prompt = f"""
{JUDGE_INSTRUCTIONS}

USER QUESTION:
{question}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}
"""

    result = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )

    verdict = parse_json_response(result.text)

    if verdict.get("winner") not in {"A", "B", "tie"}:
        raise ValueError("Judge returned an invalid winner.")

    for label in ("A", "B"):
        scores = verdict.get("scores", {}).get(label, {})

        for dimension in (
            "correctness",
            "relevance",
            "instruction_following",
            "clarity",
        ):
            score = scores.get(dimension)

            if not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(
                    f"Invalid {dimension} score for response {label}."
                )

    return verdict


def reverse_verdict(verdict):
    """Map a reversed-order verdict back to original A/B labels."""
    reversed_scores = verdict["scores"]

    return {
        "scores": {
            "A": reversed_scores["B"],
            "B": reversed_scores["A"],
        },
        "winner": {
            "A": "B",
            "B": "A",
            "tie": "tie",
        }[verdict["winner"]],
        "reason": verdict.get("reason", ""),
    }


def average_score(scores):
    dimensions = (
        "correctness",
        "relevance",
        "instruction_following",
        "clarity",
    )

    return sum(scores[d] for d in dimensions) / len(dimensions)


def main():
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Judge dataset not found: {CASES_PATH}"
        )

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    results = []
    position_changes = 0
    successful_cases = 0

    for case in cases:
        case_id = case.get("id", "unknown")
        print(f"\nEvaluating {case_id}...")

        started = time.perf_counter()

        try:
            # First run: original order.
            forward = judge_pair(
                case["question"],
                case["response_a"],
                case["response_b"],
            )

            # Second run: reverse the response positions.
            reverse_raw = judge_pair(
                case["question"],
                case["response_b"],
                case["response_a"],
            )

            # Normalize the reversed verdict to the original labels.
            reverse = reverse_verdict(reverse_raw)

            position_changed = (
                forward["winner"] != reverse["winner"]
            )

            if position_changed:
                position_changes += 1

            successful_cases += 1

            result = {
                "id": case_id,
                "task": case.get("task", "unspecified"),
                "question": case["question"],
                "expected_preference": case.get(
                    "expected_preference"
                ),
                "forward_verdict": forward,
                "reverse_normalized_verdict": reverse,
                "position_changed_preference": position_changed,
                "forward_mean_score_A": average_score(
                    forward["scores"]["A"]
                ),
                "forward_mean_score_B": average_score(
                    forward["scores"]["B"]
                ),
                "latency_seconds": round(
                    time.perf_counter() - started, 2
                ),
                "error": None,
            }

            print("Forward winner:", forward["winner"])
            print("Reversed-order winner:", reverse["winner"])
            print("Preference changed:", position_changed)

        except Exception as exc:
            logger.exception("Failed to evaluate case %s", case_id)

            result = {
                "id": case_id,
                "question": case.get("question", ""),
                "error": str(exc),
                "latency_seconds": round(
                    time.perf_counter() - started, 2
                ),
            }

        results.append(result)

    report = {
        "judge_model": GEMINI_MODEL,
        "cases_loaded": len(cases),
        "cases_successfully_evaluated": successful_cases,
        "cases_failed": len(cases) - successful_cases,
        "judge_calls_per_successful_case": 2,
        "position_preference_changes": position_changes,
        "position_preference_change_rate": (
            position_changes / successful_cases
            if successful_cases else None
        ),
        "results": results,
        "limitations": [
            "The judge is itself an LLM and can make mistakes.",
            "Position preference changes are a diagnostic, not proof of bias.",
            "Only a small set of cases is evaluated.",
            "Expected preferences are reference labels and may need human review.",
            "This script does not independently measure sycophancy, "
            "self-enhancement, verbosity bias, or score clustering.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(
        {key: value for key, value in report.items()
         if key != "results"},
        indent=2,
    ))
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
