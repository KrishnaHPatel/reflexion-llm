"""
Evaluator for CoT, ReAct, and Reflexion QA experiments.

Provides normalized exact match and token-level F1, following the style of
SQuAD/HotpotQA-style answer evaluation.
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


def extract_answer(response: str) -> str:
    """
    Extract the final answer from a model response.

    Preferred formats:
      ANSWER: <answer>
      Finish[<answer>]

    Falls back to the final non-empty line.
    """
    if not response:
        return ""

    finish_matches = re.findall(r"Finish\[(.*?)\]", response, flags=re.IGNORECASE | re.DOTALL)
    if finish_matches:
        return finish_matches[-1].strip()

    answer_matches = re.findall(r"ANSWER\s*:\s*(.+?)(?:\n|$)", response, flags=re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].strip()

    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else response.strip()


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
