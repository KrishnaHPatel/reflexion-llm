"""
Evaluator for Reflexion

This module implements exact-match evaluation with normalization.
The evaluator compares the model's answer to the gold answer after
applying normalization (lowercasing, stripping whitespace, etc.).
"""

import re
import string


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Applies the following normalizations:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace
    
    This follows standard QA evaluation practices from datasets like SQuAD.
    
    Args:
        answer: The raw answer string
    
    Returns:
        The normalized answer string
    """
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    
    # Remove articles
    articles = ["a", "an", "the"]
    words = answer.split()
    words = [w for w in words if w not in articles]
    answer = " ".join(words)
    
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    return answer


def extract_answer(response: str) -> str:
    """
    Extract the final answer from a model response.
    
    Looks for the pattern "ANSWER: <answer>" at the end of the response.
    If not found, returns the last non-empty line as a fallback.
    
    Args:
        response: The full model response with reasoning
    
    Returns:
        The extracted answer string
    """
    # Try to find explicit "ANSWER:" pattern
    match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: return the last non-empty line
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if lines:
        return lines[-1]
    
    return response.strip()


def evaluate(model_answer: str, gold_answer: str) -> bool:
    """
    Evaluate if the model's answer matches the gold answer.
    
    Uses exact match after normalization. This is a standard evaluation
    approach for QA tasks as described in the Reflexion paper for HotPotQA.
    
    Args:
        model_answer: The answer extracted from the model's response
        gold_answer: The correct answer from the dataset
    
    Returns:
        True if answers match after normalization, False otherwise
    """
    normalized_model = normalize_answer(model_answer)
    normalized_gold = normalize_answer(gold_answer)
    
    return normalized_model == normalized_gold
