"""
Prompt templates for CoT, ReAct, and Reflexion QA experiments.

The reflection prompt intentionally does NOT reveal the gold answer. It only
provides the failed trajectory and the binary feedback signal, matching the
Reflexion idea of turning sparse feedback into verbal lessons.
"""

from __future__ import annotations


COT_SYSTEM_PROMPT = """You are a careful question-answering assistant.
Use concise step-by-step reasoning, then end with exactly one final line:
ANSWER: <short answer>"""


REACT_SYSTEM_PROMPT = """You are a ReAct question-answering agent.
You solve questions by alternating brief reasoning with exactly one action.
Available actions:
- Search[query]: search the provided context/retrieval environment.
- Lookup[string]: look up a specific string in the current or provided context.
- Finish[answer]: submit the final short answer.

At every step, respond with exactly this format:
Thought: <brief reasoning about what to do next>
Action: <Search[...] or Lookup[...] or Finish[...]>

Do not answer outside the Action field."""


REFLECTOR_SYSTEM_PROMPT = """You are a self-reflection module for a QA agent.
Given a failed attempt and only a failure signal, write a concise first-person
lesson that can help the agent avoid the same mistake next trial.
Do not invent the gold answer. Focus on strategy: what to inspect, compare,
search, or reason about differently."""


COT_FEW_SHOT = """Example 1:
Question: What profession does John Lanchester and Alan Dean Foster have in common?
Context:
John Lanchester is a British novelist, journalist, and critic. Alan Dean Foster is an American novelist and screenwriter.
Reasoning: John Lanchester is listed as a novelist. Alan Dean Foster is also listed as a novelist. The common profession is novelist.
ANSWER: novelist

Example 2:
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Context:
Arthur's Magazine was an American literary periodical published from 1844 to 1846. First for Women is a magazine started in 1989.
Reasoning: Arthur's Magazine started in 1844. First for Women started in 1989. Since 1844 is earlier than 1989, Arthur's Magazine came first.
ANSWER: Arthur's Magazine
"""


REACT_FEW_SHOT = """Example ReAct trajectory:
Question: What profession does John Lanchester and Alan Dean Foster have in common?
Thought: I need to find each person's profession and compare them.
Action: Search[John Lanchester Alan Dean Foster profession]
Observation: John Lanchester is a British novelist, journalist, and critic. Alan Dean Foster is an American novelist and screenwriter.
Thought: Both people are described as novelists, so the shared profession is novelist.
Action: Finish[novelist]

Example ReAct trajectory:
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Thought: I need the start dates for both magazines.
Action: Search[Arthur's Magazine First for Women started]
Observation: Arthur's Magazine was published from 1844 to 1846. First for Women was started in 1989.
Thought: 1844 is earlier than 1989, so Arthur's Magazine started first.
Action: Finish[Arthur's Magazine]
"""


REFLECTION_FEW_SHOT = """Example failed attempt:
Question: Which of two people is older?
Trajectory: I only checked one birth year and guessed the other.
Feedback: Answer is INCORRECT.
Reflection: I failed because I did not verify both entities before comparing them. Next time, I should explicitly find evidence for each entity, compare the relevant values, and only then finish.

Example failed attempt:
Question: What role was an actor best known for?
Trajectory: I searched the show title, found a different actor, and finished with that actor's role.
Feedback: Answer is INCORRECT.
Reflection: I likely followed the wrong entity. Next time, I should identify the actor from the question/context first, then search or inspect evidence for that specific actor's role before answering.
"""


def _format_memory(memory: list[str] | None) -> str:
    if not memory:
        return ""
    lines = ["Reflections from previous failed trials:"]
    for i, reflection in enumerate(memory, 1):
        lines.append(f"{i}. {reflection}")
    return "\n".join(lines) + "\n\nUse these reflections, but still answer from the evidence.\n\n"


def build_cot_prompt(question: str, context: str = "", memory: list[str] | None = None, include_few_shot: bool = True) -> str:
    parts = []
    if include_few_shot:
        parts.append(COT_FEW_SHOT.strip() + "\n\n")
    parts.append(_format_memory(memory))
    if context:
        parts.append(f"Context:\n{context}\n\n")
    parts.append(f"Question: {question}\n")
    parts.append("Reason step by step, then provide the final answer as `ANSWER: <short answer>`.")
    return "".join(parts)


def build_react_prompt(
    question: str,
    trajectory: str = "",
    memory: list[str] | None = None,
    include_few_shot: bool = True,
) -> str:
    parts = []
    if include_few_shot:
        parts.append(REACT_FEW_SHOT.strip() + "\n\n")
    parts.append(_format_memory(memory))
    parts.append(f"Question: {question}\n")
    if trajectory:
        parts.append(trajectory.rstrip() + "\n")
    parts.append("Next step:")
    return "".join(parts)


def build_reflector_prompt(
    question: str,
    trajectory: str,
    model_answer: str,
    feedback: str = "Answer is INCORRECT.",
    context: str = "",
    include_few_shot: bool = True,
) -> str:
    parts = []
    if include_few_shot:
        parts.append(REFLECTION_FEW_SHOT.strip() + "\n\n")
    parts.append("A QA attempt failed. Reflect on the failed attempt without using the gold answer.\n\n")
    parts.append(f"Question: {question}\n\n")
    if context:
        parts.append(f"Available context/retrieval source:\n{context}\n\n")
    parts.append(f"Failed trajectory or reasoning:\n{trajectory}\n\n")
    parts.append(f"Submitted answer: {model_answer}\n")
    parts.append(f"Feedback: {feedback}\n\n")
    parts.append("Write one concise first-person reflection with an actionable strategy for the next trial.")
    return "".join(parts)


# Backward-compatible aliases for your original code names.
ACTOR_SYSTEM_PROMPT = COT_SYSTEM_PROMPT

def build_actor_prompt(question: str, context: str = "", memory: list[str] | None = None) -> str:
    return build_cot_prompt(question=question, context=context, memory=memory)
