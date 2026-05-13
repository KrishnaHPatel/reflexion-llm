"""
Evaluator for CoT, ReAct, and Reflexion QA experiments.

Provides robust answer extraction, normalized exact match, and token-level F1,
following the style of SQuAD/HotpotQA-style answer evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import re
import string


@dataclass
class AnswerEvaluation:
    """Serializable answer-evaluation result."""

    exact_match: bool
    f1: float
    normalized_model_answer: str
    normalized_gold_answer: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Steps:
    1. lowercase
    2. remove punctuation
    3. remove English articles: a, an, the
    4. collapse whitespace
    """
    if answer is None:
        answer = ""

    answer = str(answer).lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in answer.split() if w not in {"a", "an", "the"}]
    return " ".join(words)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 after normalization."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def _clean_candidate(candidate: str) -> str:
    """Clean a candidate extracted answer."""
    if candidate is None:
        return ""

    candidate = str(candidate).strip()

    # Remove wrapping quotes/backticks.
    candidate = candidate.strip("`").strip()
    candidate = candidate.strip('"').strip("'").strip()

    # Remove trailing markdown/code artifacts.
    candidate = re.sub(r"\s*```.*$", "", candidate).strip()

    # If model writes "ANSWER: X" inside another extraction path.
    candidate = re.sub(r"^(final\s+answer|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE).strip()

    # Remove a trailing period for short answers, but keep internal punctuation.
    if len(candidate.split()) <= 8:
        candidate = candidate.rstrip(".").strip()

    return candidate


def _looks_like_bad_answer(candidate: str) -> bool:
    """
    Return True if the candidate is probably not an answer.

    This prevents things like "Thought: I need to search..." or
    "Fix: verify both entities" from being used as final answers.
    """
    if not candidate or not candidate.strip():
        return True

    low = candidate.strip().lower()

    bad_prefixes = (
        "thought:",
        "action:",
        "observation:",
        "reflection:",
        "error:",
        "cause:",
        "fix:",
        "feedback:",
        "trajectory:",
        "question:",
        "context:",
        "reasoning:",
        "next step:",
        "submitted answer:",
        "available context",
        "failed trajectory",
    )

    if low.startswith(bad_prefixes):
        return True

    # Blank answer markers.
    if low in {
        "answer:",
        "final answer:",
        "final:",
        "finish[]",
        "finish[ ]",
        "action: finish[]",
    }:
        return True

    # Template text accidentally copied.
    if "<short answer>" in low or "<short final answer>" in low or "<answer>" in low:
        return True

    # Model says it cannot answer instead of answering.
    non_answer_phrases = (
        "i don't know",
        "i do not know",
        "cannot determine",
        "can't determine",
        "not enough information",
        "insufficient information",
        "unknown",
    )

    # Keep "not enough information" if it is actually the gold label style in some datasets,
    # but for HotpotQA it is usually not a final answer. You can remove this block if needed.
    if low in non_answer_phrases:
        return True

    return False


def _extract_yes_no_from_text(response: str) -> str:
    """
    Extract yes/no when the model answers semantically but not in the exact format.

    Examples:
    - "They are both American, so yes." -> yes
    - "No, they were not the same nationality." -> no
    """
    text = response.strip().lower()

    # Prefer explicit final-ish yes/no statements near the end.
    tail = " ".join(text.split()[-40:])

    yes_patterns = [
        r"\bso\s+yes\b",
        r"\btherefore\s+yes\b",
        r"\bthe\s+answer\s+is\s+yes\b",
        r"\banswer\s*:\s*yes\b",
        r"\bfinal\s+answer\s*:\s*yes\b",
        r"\bfinish\[yes\]",
        r"\byes[,.\s]*$",
    ]

    no_patterns = [
        r"\bso\s+no\b",
        r"\btherefore\s+no\b",
        r"\bthe\s+answer\s+is\s+no\b",
        r"\banswer\s*:\s*no\b",
        r"\bfinal\s+answer\s*:\s*no\b",
        r"\bfinish\[no\]",
        r"\bno[,.\s]*$",
    ]

    for pat in yes_patterns:
        if re.search(pat, tail, flags=re.IGNORECASE):
            return "yes"

    for pat in no_patterns:
        if re.search(pat, tail, flags=re.IGNORECASE):
            return "no"

    return ""


def extract_answer(response: str) -> str:
    """
    Extract the final answer from a model response.

    Supports:
    - ANSWER: <answer>
    - Final answer: <answer>
    - Action: Finish[<answer>]
    - Finish[<answer>]

    Falls back to the final non-empty line that does not look like a thought,
    reflection, action, or template artifact.
    """
    if not response:
        return ""

    response = str(response).strip()
    if not response:
        return ""

    # 1. ReAct final action: take the LAST non-empty Finish[...] if present.
    finish_matches = re.findall(
        r"Finish\s*\[\s*(.*?)\s*\]",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    finish_candidates = [_clean_candidate(x) for x in finish_matches]
    finish_candidates = [x for x in finish_candidates if not _looks_like_bad_answer(x)]
    if finish_candidates:
        return finish_candidates[-1]

    # 2. Strict ANSWER: line. Take the LAST non-empty one.
    answer_matches = re.findall(
        r"^\s*ANSWER\s*:\s*(.*?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    answer_candidates = [_clean_candidate(x) for x in answer_matches]
    answer_candidates = [x for x in answer_candidates if not _looks_like_bad_answer(x)]
    if answer_candidates:
        return answer_candidates[-1]

    # 3. Common variant: Final answer: line.
    final_answer_matches = re.findall(
        r"^\s*FINAL\s+ANSWER\s*:\s*(.*?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    final_candidates = [_clean_candidate(x) for x in final_answer_matches]
    final_candidates = [x for x in final_candidates if not _looks_like_bad_answer(x)]
    if final_candidates:
        return final_candidates[-1]

    # 4. Handle cases where model writes:
    #    "Final answer:\nParis"
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    for i, line in enumerate(lines[:-1]):
        low = line.lower().strip()
        if low in {"answer:", "final answer:", "final:"}:
            candidate = _clean_candidate(lines[i + 1])
            if not _looks_like_bad_answer(candidate):
                return candidate

    # 5. If response clearly contains a final yes/no in prose, extract it.
    yn = _extract_yes_no_from_text(response)
    if yn:
        return yn

    # 6. Last-line fallback, skipping non-answer lines.
    for line in reversed(lines):
        candidate = _clean_candidate(line)

        # If line is "Action: Finish[X]", extract X.
        action_finish = re.search(
            r"Action\s*:\s*Finish\s*\[\s*(.*?)\s*\]",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if action_finish:
            candidate = _clean_candidate(action_finish.group(1))

        if not _looks_like_bad_answer(candidate):
            return candidate

    return ""


def evaluate_answer(model_answer: str, gold_answer: str) -> AnswerEvaluation:
    """Return exact-match and token-F1 evaluation for one answer."""
    normalized_model = normalize_answer(model_answer)
    normalized_gold = normalize_answer(gold_answer)

    return AnswerEvaluation(
        exact_match=normalized_model == normalized_gold,
        f1=token_f1(model_answer, gold_answer),
        normalized_model_answer=normalized_model,
        normalized_gold_answer=normalized_gold,
    )


def evaluate(model_answer: str, gold_answer: str) -> bool:
    """Backward-compatible exact-match boolean evaluator."""
    return evaluate_answer(model_answer, gold_answer).exact_match