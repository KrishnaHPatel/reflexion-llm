# Reflexion Reasoning Baseline

A minimal Python implementation of the Reflexion paper's reasoning baseline (Shinn et al., NeurIPS 2023).

## Overview

Reflexion enables language models to learn from mistakes through **verbal reinforcement** - storing natural language reflections in memory to improve subsequent attempts, without any weight updates.

### Components

- **Actor**: Generates answers using Chain-of-Thought reasoning, conditioned on past reflections
- **Evaluator**: Compares model answers to gold answers using exact match (after normalization)
- **Reflector**: Analyzes failed attempts and generates actionable reflections
- **Memory**: Stores up to 3 reflections to guide future trials

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Ollama (Free, Local)

1. Install Ollama from https://ollama.ai
2. Pull a model:
```bash
ollama pull llama3.2
```
3. Run (Ollama is the default):
```bash
python main.py --input data/sample.jsonl --output results.jsonl
```

### Option 2: Groq (Free Tier)

1. Get a free API key at https://console.groq.com
2. Set the key and run:
```bash
export GROQ_API_KEY="your-key"
python main.py --provider groq
```

### Option 3: OpenAI (Paid)

```bash
export OPENAI_API_KEY="your-key"
python main.py --provider openai
```

### Command-line Options

```
--provider     LLM provider: ollama (free), groq (free tier), openai (paid)
--input        Path to input JSONL file (default: data/sample.jsonl)
--output       Path to output JSONL file (default: results.jsonl)
--model        LLM model (defaults based on provider)
--max-trials   Maximum trials per question (default: 3)
--max-memory   Maximum reflections to store (default: 3)
--temperature  Sampling temperature (default: 0.7)
```

## Input Format

JSONL file with one question per line:
```json
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Who wrote Hamlet?", "answer": "Shakespeare", "context": "Optional context..."}
```

## Output Format

JSONL file with detailed results for each question:
```json
{
  "question": "...",
  "gold_answer": "...",
  "final_answer": "...",
  "solved": true,
  "num_trials": 2,
  "trials": [
    {
      "trial_number": 1,
      "model_answer": "...",
      "is_correct": false,
      "reasoning": "...",
      "reflection": "..."
    },
    {
      "trial_number": 2,
      "model_answer": "...",
      "is_correct": true,
      "reasoning": "...",
      "reflection": ""
    }
  ],
  "memory": ["reflection from trial 1"]
}
```

## Project Structure

```
reflexion-llm/
├── main.py          # Entry point - orchestrates the Reflexion loop
├── llm.py           # LLM API wrapper (OpenAI-compatible)
├── prompts.py       # Prompt templates for Actor and Reflector
├── evaluator.py     # Exact-match evaluation with normalization
├── reflexion.py     # Core Reflexion agent implementation
├── data/
│   └── sample.jsonl # Sample questions for testing
├── requirements.txt
└── README.md
```

## Reference

```
@inproceedings{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and Cassano, Federico and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu},
  booktitle={NeurIPS},
  year={2023}
}
```
