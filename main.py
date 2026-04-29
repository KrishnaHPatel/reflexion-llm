"""
Reflexion Reasoning Baseline - Main Entry Point

This script runs the Reflexion agent on a dataset of questions.
It loads questions from a JSONL file, runs the Reflexion loop on each,
and saves detailed logs of the results.

Usage:
    python main.py --input data/sample.jsonl --output results.jsonl

The Reflexion approach (Shinn et al., 2023) enables LLMs to learn from
mistakes through natural language reflection, without any weight updates.
"""

import argparse
import json
from pathlib import Path

from reflexion import ReflexionAgent


def load_data(path: str) -> list[dict]:
    """
    Load questions from a JSONL file.
    
    Expected format per line:
    {"question": "...", "answer": "...", "context": "..." (optional)}
    
    Args:
        path: Path to the JSONL file
    
    Returns:
        List of question dictionaries
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_results(results: list[dict], path: str) -> None:
    """
    Save results to a JSONL file.
    
    Each line contains the full result for one question, including
    all trial attempts, reflections, and memory state.
    
    Args:
        results: List of result dictionaries
        path: Output path for the JSONL file
    """
    with open(path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def result_to_dict(result) -> dict:
    """
    Convert a ReflexionResult object to a serializable dictionary.
    
    The output format includes all information needed for analysis:
    - question: The original question
    - gold_answer: The correct answer
    - trials: List of trial attempts with reasoning and reflections
    - solved: Whether the question was answered correctly
    - memory: Final state of the reflection memory
    """
    trials_data = []
    for trial in result.trials:
        trials_data.append({
            "trial_number": trial.trial_number,
            "model_answer": trial.model_answer,
            "is_correct": trial.is_correct,
            "reasoning": trial.reasoning,
            "reflection": trial.reflection,
        })
    
    return {
        "question": result.question,
        "gold_answer": result.gold_answer,
        "context": result.context,
        "final_answer": result.final_answer,
        "solved": result.solved,
        "num_trials": len(result.trials),
        "trials": trials_data,
        "memory": result.memory,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Reflexion reasoning baseline on a QA dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample.jsonl",
        help="Path to input JSONL file with questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.jsonl",
        help="Path to output JSONL file for results",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "groq", "openai"],
        help="LLM provider: ollama (free), groq (free tier), openai (paid)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model (defaults based on provider)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="Maximum trials per question (default: 3)",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=3,
        help="Maximum reflections to keep in memory (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}", flush=True)
    data = load_data(args.input)
    print(f"Loaded {len(data)} questions", flush=True)
    
    # Initialize agent
    print(f"Using provider: {args.provider}", flush=True)
    agent = ReflexionAgent(
        model=args.model,
        provider=args.provider,
        max_trials=args.max_trials,
        max_memory=args.max_memory,
        temperature=args.temperature,
    )
    
    # Run Reflexion on each question
    results = []
    correct_first_try = 0
    correct_after_reflection = 0
    
    for i, item in enumerate(data):
        question = item["question"]
        answer = item["answer"]
        context = item.get("context", "")
        
        print(f"\n[{i+1}/{len(data)}] Processing: {question[:50]}...", flush=True)
        
        result = agent.run(
            question=question,
            gold_answer=answer,
            context=context,
        )
        
        # Track statistics
        if result.solved:
            if len(result.trials) == 1:
                correct_first_try += 1
                print(f"  ✓ Correct on first try", flush=True)
            else:
                correct_after_reflection += 1
                print(f"  ✓ Correct after {len(result.trials)} trials (reflection helped!)", flush=True)
        else:
            print(f"  ✗ Failed after {len(result.trials)} trials", flush=True)
        
        results.append(result_to_dict(result))
    
    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    
    # Print summary statistics
    total = len(data)
    total_correct = correct_first_try + correct_after_reflection
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total questions: {total}")
    print(f"Correct on first try: {correct_first_try} ({100*correct_first_try/total:.1f}%)")
    print(f"Correct after reflection: {correct_after_reflection} ({100*correct_after_reflection/total:.1f}%)")
    print(f"Total correct: {total_correct} ({100*total_correct/total:.1f}%)")
    print(f"Reflection improvement: +{correct_after_reflection} questions")


if __name__ == "__main__":
    main()
