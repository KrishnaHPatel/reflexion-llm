"""
Analyze JSONL outputs from main.py.

Usage:
    # Analyze all default experiment outputs
    python analyze_results.py

    # Analyze specific files only
    python analyze_results.py results_reflexion.jsonl
    python analyze_results.py results_cot.jsonl results_reflexion.jsonl results_react_reflexion.jsonl

    # Show failures too
    python analyze_results.py --show-failures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_RESULT_FILES = [
    "results_cot.jsonl",
    "results_react.jsonl",
    "results_react_compressed.jsonl",
    "results_react_policy.jsonl",
    "results_react_reflexion.jsonl",
    "results_react_structured.jsonl",
    "results_reflexion.jsonl",
]


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    total = len(rows)

    first_try = sum(1 for r in rows if r.get("first_try_correct"))
    solved_after = sum(1 for r in rows if r.get("solved_after_first_try"))
    solved = sum(1 for r in rows if r.get("solved"))
    failed = total - solved

    avg_trials = sum(r.get("num_trials", 0) for r in rows) / total if total else 0.0
    avg_best_f1 = sum(r.get("best_f1", 0.0) for r in rows) / total if total else 0.0

    all_trial_f1 = []
    for row in rows:
        for trial in row.get("trials", []):
            all_trial_f1.append(trial.get("evaluation", {}).get("f1", 0.0))

    avg_trial_f1 = sum(all_trial_f1) / len(all_trial_f1) if all_trial_f1 else 0.0

    return {
        "total": total,
        "first_try_correct": first_try,
        "solved_after_first_try": solved_after,
        "total_solved": solved,
        "failed": failed,
        "first_try_em": first_try / total if total else 0.0,
        "final_em": solved / total if total else 0.0,
        "gain_over_first_try": (solved - first_try) / total if total else 0.0,
        "avg_trials": avg_trials,
        "avg_best_f1": avg_best_f1,
        "avg_trial_f1": avg_trial_f1,
    }


def print_table(summaries: list[tuple[str, dict]]) -> None:
    headers = [
        "file",
        "n",
        "first_EM",
        "final_EM",
        "gain",
        "avg_best_F1",
        "avg_trial_F1",
        "avg_trials",
        "failed",
    ]

    print("\t".join(headers))

    for name, s in summaries:
        print("\t".join([
            name,
            str(s["total"]),
            f"{s['first_try_em']:.3f}",
            f"{s['final_em']:.3f}",
            f"{s['gain_over_first_try']:.3f}",
            f"{s['avg_best_f1']:.3f}",
            f"{s['avg_trial_f1']:.3f}",
            f"{s['avg_trials']:.2f}",
            str(s["failed"]),
        ]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze QA agent result JSONL files")

    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more result JSONL files. If omitted, all default result files are analyzed.",
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failed questions and final answers",
    )

    args = parser.parse_args()

    paths = args.paths if args.paths else DEFAULT_RESULT_FILES

    summaries = []
    all_rows_by_path = {}

    for path in paths:
        path_obj = Path(path)

        if not path_obj.exists():
            print(f"Warning: skipping missing file: {path}")
            continue

        rows = load_results(path)
        summaries.append((path_obj.name, summarize(rows)))
        all_rows_by_path[path] = rows

    if not summaries:
        print("No result files found.")
        return

    print_table(summaries)

    if args.show_failures:
        for path, rows in all_rows_by_path.items():
            print(f"\nFailures in {path}:")
            for row in rows:
                if not row.get("solved"):
                    print(f"- Q: {row.get('question')}")
                    print(f"  Gold: {row.get('gold_answer')}")
                    print(f"  Final: {row.get('final_answer')}")
                    print(f"  Best F1: {row.get('best_f1', 0.0):.3f}")


if __name__ == "__main__":
    main()