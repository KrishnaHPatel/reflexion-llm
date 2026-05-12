#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original JSONL file")
    parser.add_argument("--output", required=True, help="Sampled output JSONL file")
    parser.add_argument("--n", type=int, default=100, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = [
        line for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.n > len(lines):
        raise ValueError(f"Requested {args.n} examples, but file only has {len(lines)} lines.")

    random.seed(args.seed)
    sampled = random.sample(lines, args.n)

    output_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")

    print(f"Loaded {len(lines)} examples")
    print(f"Sampled {len(sampled)} examples")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()