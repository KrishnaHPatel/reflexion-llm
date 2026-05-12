#!/usr/bin/env python3
"""
Convert HotpotQA-style JSON / JSONL into simple QA JSONL.

Input format:
[
  {
    "_id": "...",
    "question": "...",
    "answer": "...",
    "context": [
      ["Title 1", ["sentence 1", "sentence 2"]],
      ["Title 2", ["sentence 1", "sentence 2"]]
    ],
    ...
  }
]

Output JSONL format:
{"question": "...", "answer": "...", "context": "Title 1: sentence sentence ... Title 2: sentence sentence ..."}
"""

import argparse
import json
from pathlib import Path


def load_input(path: Path):
    """
    Load either:
    - a JSON file containing a list of examples
    - a JSONL file containing one JSON object per line
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # Try normal JSON first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    examples = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    return examples


def flatten_context(context, only_supporting_titles=None):
    """
    Convert HotpotQA context list into one plain text string.

    context format:
    [
      ["Article Title", ["sentence 1", "sentence 2"]],
      ...
    ]

    If only_supporting_titles is provided, only those article titles are kept.
    """
    parts = []

    for item in context:
        if not isinstance(item, list) or len(item) != 2:
            continue

        title, sentences = item

        if only_supporting_titles is not None and title not in only_supporting_titles:
            continue

        if isinstance(sentences, list):
            passage = " ".join(s.strip() for s in sentences if isinstance(s, str))
        elif isinstance(sentences, str):
            passage = sentences.strip()
        else:
            continue

        if passage:
            parts.append(f"{title}: {passage}")

    return "\n\n".join(parts)


def convert_example(example, use_supporting_only=False):
    question = example.get("question", "").strip()
    answer = str(example.get("answer", "")).strip()

    context = example.get("context", [])

    supporting_titles = None
    if use_supporting_only:
        supporting_facts = example.get("supporting_facts", [])
        supporting_titles = {
            fact[0]
            for fact in supporting_facts
            if isinstance(fact, list) and len(fact) >= 1
        }

    flat_context = flatten_context(
        context,
        only_supporting_titles=supporting_titles,
    )

    return {
        "question": question,
        "answer": answer,
        "context": flat_context,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert HotpotQA JSON/JSONL to simple QA JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to HotpotQA-style input JSON or JSONL file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--supporting-only",
        action="store_true",
        help="Only include articles listed in supporting_facts.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    examples = load_input(input_path)

    converted = [
        convert_example(example, use_supporting_only=args.supporting_only)
        for example in examples
    ]

    with output_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} examples")
    print(f"Wrote output to {output_path}")


if __name__ == "__main__":
    main()