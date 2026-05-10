"""
Main entry point for CoT, Reflexion, ReAct, and ReAct+Reflexion QA runs.

Usage examples:
    python3 main.py --input data/hotpotqa_sample.jsonl --output results_cot.jsonl --mode cot
    python3 main.py --input data/hotpotqa_sample.jsonl --output results_reflexion.jsonl --mode reflexion
    python3 main.py --input data/hotpotqa_sample.jsonl --output results_react.jsonl --mode react
    python3 main.py --input data/hotpotqa_sample.jsonl --output results_react_reflexion.jsonl --mode react-reflexion

This script does not implement the coding/programming Reflexion pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexion import ReflexionAgent, ReflexionResult


def load_data(path: str) -> list[dict]:
    """Load JSONL QA data with fields: question, answer, optional context."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "question" not in item or "answer" not in item:
                raise ValueError(f"Line {line_num} must contain 'question' and 'answer' fields.")
            data.append(item)
    return data


def save_results(results: list[dict], path: str) -> None:
    """Save JSONL results."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def result_to_dict(result: ReflexionResult) -> dict:
    """Convert a ReflexionResult object into a JSON-serializable dict."""
    trials_data = []
    for trial in result.trials:
        trials_data.append(
            {
                "trial_number": trial.trial_number,
                "model_answer": trial.model_answer,
                "is_correct": trial.is_correct,
                "evaluation": trial.evaluation.to_dict(),
                "reasoning": trial.reasoning,
                "reflection": trial.reflection,
                "trajectory": trial.trajectory,
            }
        )

    best_f1 = max((t.evaluation.f1 for t in result.trials), default=0.0)
    first_trial = result.trials[0] if result.trials else None

    return {
        "question": result.question,
        "gold_answer": result.gold_answer,
        "context": result.context,
        "mode": result.mode,
        "reflection_type": result.reflection_type,
        "final_answer": result.final_answer,
        "solved": result.solved,
        "num_trials": len(result.trials),
        "first_try_correct": bool(first_trial and first_trial.is_correct),
        "solved_after_first_try": bool(result.solved and first_trial and not first_trial.is_correct),
        "best_f1": best_f1,
        "trials": trials_data,
        "memory": result.memory,
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)
    first_try = sum(1 for r in results if r.get("first_try_correct"))
    after_first = sum(1 for r in results if r.get("solved_after_first_try"))
    solved = sum(1 for r in results if r.get("solved"))
    failed = total - solved
    avg_trials = sum(r.get("num_trials", 0) for r in results) / total if total else 0.0
    avg_best_f1 = sum(r.get("best_f1", 0.0) for r in results) / total if total else 0.0

    return {
        "total": total,
        "first_try_correct": first_try,
        "solved_after_first_try": after_first,
        "total_solved": solved,
        "failed": failed,
        "first_try_em": first_try / total if total else 0.0,
        "final_em": solved / total if total else 0.0,
        "reflection_or_retry_gain": after_first / total if total else 0.0,
        "avg_trials": avg_trials,
        "avg_best_f1": avg_best_f1,
    }


def print_summary(summary: dict) -> None:
    total = summary["total"] or 1
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions: {summary['total']}")
    print(f"Correct on first try: {summary['first_try_correct']} ({100*summary['first_try_correct']/total:.1f}%)")
    print(f"Solved after first try: {summary['solved_after_first_try']} ({100*summary['solved_after_first_try']/total:.1f}%)")
    print(f"Total solved EM: {summary['total_solved']} ({100*summary['final_em']:.1f}%)")
    print(f"Failed: {summary['failed']} ({100*summary['failed']/total:.1f}%)")
    print(f"Average trials: {summary['avg_trials']:.2f}")
    print(f"Average best token F1: {summary['avg_best_f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA agents: CoT, Reflexion, ReAct, ReAct+Reflexion")
    parser.add_argument("--input", type=str, default="data/sample.jsonl", help="Input JSONL with question, answer, optional context")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cot", "reflexion", "react", "react-reflexion"],
        default="reflexion",
        help="Agent mode to run",
    )
    parser.add_argument("--model", type=str, default=None, help="Ollama model name, e.g. llama3.2 or mistral")
    parser.add_argument(
        "--reflection-type",
        type=str,
        choices=["freeform", "structured", "policy", "compressed"],
        default="freeform",
        help="Reflection memory format for reflexion modes",
    )
    parser.add_argument("--max-trials", type=int, default=3, help="Maximum trials per question")
    parser.add_argument("--max-memory", type=int, default=3, help="Maximum reflections in memory")
    parser.add_argument("--max-react-steps", type=int, default=6, help="Maximum Search/Lookup/Finish steps per ReAct trial")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generated tokens per LLM call")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable built-in few-shot examples")
    args = parser.parse_args()

    print(f"Loading data from {args.input}", flush=True)
    data = load_data(args.input)
    print(f"Loaded {len(data)} questions", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Reflection type: {args.reflection_type}", flush=True)
    print(f"Using Ollama model: {args.model or 'llama3.2'}", flush=True)

    agent = ReflexionAgent(
        model=args.model,
        mode=args.mode,
        reflection_type=args.reflection_type,
        max_trials=args.max_trials,
        max_memory=args.max_memory,
        max_react_steps=args.max_react_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_few_shot=not args.no_few_shot,
    )

    results = []
    for i, item in enumerate(data, 1):
        question = item["question"]
        answer = item["answer"]
        context = item.get("context", "")
        print(f"\n[{i}/{len(data)}] {question[:80]}", flush=True)

        result = agent.run(question=question, gold_answer=answer, context=context)
        result_dict = result_to_dict(result)
        results.append(result_dict)

        if result_dict["solved"]:
            where = "first try" if result_dict["first_try_correct"] else f"trial {result_dict['num_trials']}"
            print(f"  ✓ Solved on {where}: {result_dict['final_answer']}", flush=True)
        else:
            print(f"  ✗ Failed. Final answer: {result_dict['final_answer']}", flush=True)

    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    print_summary(summarize(results))


if __name__ == "__main__":
    main()
