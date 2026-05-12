# CoT / Reflexion / ReAct QA Baseline

This is an updated implementation of the reasoning parts of the ReAct and Reflexion papers.
It intentionally does **not** implement the programming/coding Reflexion pipeline yet.

## What is implemented

- `cot`: Chain-of-Thought QA baseline.
- `reflexion`: Chain-of-Thought + verbal reflection memory after failed trials.
- `react`: ReAct-style `Search[...]`, `Lookup[...]`, `Finish[...]` loop over the example's `context` field.
- `react-reflexion`: ReAct plus verbal reflection memory after failed trials.
- Reflection formats: `freeform`, `structured` (`Error` / `Cause` / `Fix`), `policy`, and `compressed`.
- Exact match and token-level F1 evaluation.
- Built-in few-shot prompts.
- Result analyzer for comparing runs.

## Important design choice

The reflector does **not** receive the gold answer. It receives only:

- the question,
- the failed reasoning or trajectory,
- the submitted answer,
- binary feedback: `Answer is INCORRECT.`

This avoids leaking the correct answer into the next trial.

## Files

```text
main.py                    # CLI runner
reflexion.py               # unified agent implementation
react_env.py               # offline Search/Lookup/Finish environment over context
prompts.py                 # CoT, ReAct, and reflection prompts
evaluator.py               # exact match + token F1
llm.py                     # Ollama HTTP wrapper
analyze_results.py         # result analysis script
data/sample.jsonl          # simple examples
data/hotpotqa_sample.jsonl # context-rich examples
requirements.txt           # no external deps
```

## Setup

Install Ollama and pull a model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running locally.

## Stage 1: Reproduction Runs

CoT baseline:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_cot.jsonl --mode cot
```

CoT + Reflexion:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_reflexion.jsonl --mode reflexion
```

ReAct baseline:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_react.jsonl --mode react
```

ReAct + Reflexion:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_react_reflexion.jsonl --mode react-reflexion
```

These four runs reproduce the core reasoning-agent comparisons: CoT, CoT + Reflexion, ReAct, and ReAct + Reflexion.

## Stage 2: Reflection Memory Variants

ReAct + structured Reflexion:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_react_structured.jsonl --mode react-reflexion --reflection-type structured
```

ReAct + policy Reflexion:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_react_policy.jsonl --mode react-reflexion --reflection-type policy
```

ReAct + compressed Reflexion:

```bash
python3 main.py --input data/hotpotqa_sample.jsonl --output results_react_compressed.jsonl --mode react-reflexion --reflection-type compressed
```

Recommended Stage 2 experiment matrix:

```text
Agent                  Reflection type  Purpose
ReAct                  none             Tool-use baseline
ReAct + Reflexion      freeform         Reflexion-style baseline
ReAct + Reflexion      structured       Main Error/Cause/Fix extension
ReAct + Reflexion      policy           Exploratory reusable-rule memory
ReAct + Reflexion      compressed       Exploratory token-efficient memory
```

Analyze multiple runs:

```bash
python3 analyze_results.py results_react.jsonl results_react_reflexion.jsonl results_react_structured.jsonl results_react_policy.jsonl results_react_compressed.jsonl
```

Show failures:

```bash
python3 analyze_results.py results_react_reflexion.jsonl --show-failures
```

## Input format

Each JSONL row should look like:

```json
{"question": "...", "answer": "...", "context": "optional evidence text"}
```

For ReAct modes, `context` acts as the retrieval environment. Without context, `Search[...]` and `Lookup[...]` will not have useful evidence.
