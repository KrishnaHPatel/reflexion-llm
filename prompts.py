"""
Prompt templates for CoT, ReAct, and Reflexion QA experiments.

The reflection prompt intentionally does NOT reveal the gold answer. It only
provides the failed trajectory and the binary feedback signal, matching the
Reflexion idea of turning sparse feedback into verbal lessons.
"""

from __future__ import annotations


COT_SYSTEM_PROMPT = """You are a careful question-answering assistant.

You must answer the question using the provided context.

Output format:
Reasoning: <brief reasoning>
ANSWER: <short final answer>

Rules:
- You MUST include an ANSWER line.
- The ANSWER line must not be blank.
- Do not write "Final answer:".
- Do not put explanations in the ANSWER line.
- For yes/no questions, the answer must be exactly yes or no.
"""


REACT_SYSTEM_PROMPT = """You are a ReAct question-answering agent.

You solve questions by alternating brief reasoning with exactly one action.

Available actions:
- Search[query]: search the provided context/retrieval environment.
- Lookup[string]: look up a specific string in the current or provided context.
- Finish[answer]: submit the final short answer.

At every step, respond with exactly two lines:
Thought: <brief reasoning>
Action: <Search[...] or Lookup[...] or Finish[...]>

Rules:
- You must eventually use Finish[<short answer>].
- Never use Finish[].
- For yes/no questions, use exactly Finish[yes] or Finish[no].
- Do not write "Final answer:".
- Do not answer outside the Action field.
- If you have enough evidence, finish immediately.
"""


REFLECTOR_SYSTEM_PROMPT = """You are a self-reflection module for a QA agent.

Given a failed attempt and only a failure signal, write a concise lesson that can help the agent avoid the same mistake next trial.

Rules:
- Do not answer the question.
- Do not include an ANSWER line.
- Do not invent the gold answer.
- Focus on strategy: what to inspect, compare, search, or reason about differently.
"""


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

Example 3:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Context:
Scott Derrickson is an American filmmaker. Ed Wood was an American filmmaker.
Reasoning: Scott Derrickson is American. Ed Wood was American. They had the same nationality.
ANSWER: yes

Example 4:
Question: Is the Eiffel Tower in Italy?
Context:
The Eiffel Tower is a wrought-iron tower in Paris, France.
Reasoning: The Eiffel Tower is in Paris, France, not Italy.
ANSWER: no
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

Example ReAct trajectory:
Question: Were Scott Derrickson and Ed Wood of the same nationality?
Thought: I need to find the nationality of Scott Derrickson and Ed Wood.
Action: Search[Scott Derrickson Ed Wood nationality]
Observation: Scott Derrickson is an American filmmaker. Ed Wood was an American filmmaker.
Thought: Both are American, so they are of the same nationality.
Action: Finish[yes]

Example ReAct trajectory:
Question: Is the Eiffel Tower in Italy?
Thought: I need to identify where the Eiffel Tower is located.
Action: Search[Eiffel Tower location]
Observation: The Eiffel Tower is located in Paris, France.
Thought: The Eiffel Tower is in France, not Italy.
Action: Finish[no]
"""

REACT_FINALIZER_SYSTEM_PROMPT = """You are a final-answer extractor for a QA agent.

Given a question and a ReAct trajectory, infer the best short answer from the available evidence.

Output exactly one line:
ANSWER: <short final answer>

Rules:
- The ANSWER line must not be blank.
- For yes/no questions, answer exactly yes or no.
- Do not explain.
- Do not write "Final answer:".
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

Example failed attempt:
Question: Were two people of the same nationality?
Trajectory: I wrote a sentence explaining both people but did not submit yes or no.
Feedback: Answer is INCORRECT.
Reflection: For yes/no questions, I must finish with exactly yes or no after verifying both sides from evidence.
"""


STRUCTURED_REFLECTION_FEW_SHOT = """Example failed attempt:
Question: Which of two people is older?
Trajectory: I only checked one birth year and guessed the other.
Feedback: Answer is INCORRECT.
Reflection:
Error: I guessed the comparison without verifying both entities.
Cause: I used incomplete evidence and did not explicitly compare the relevant values.
Fix: Find evidence for each entity, compare the values directly, then finish with the entity that matches the question.

Example failed attempt:
Question: What role was an actor best known for?
Trajectory: I searched the show title, found a different actor, and finished with that actor's role.
Feedback: Answer is INCORRECT.
Reflection:
Error: I followed the wrong entity.
Cause: I did not first identify the actor named or implied by the question.
Fix: Identify the relevant actor first, then search or inspect evidence for that actor's role before answering.

Example failed attempt:
Question: Were two people of the same nationality?
Trajectory: I found both people but submitted a full sentence instead of yes or no.
Feedback: Answer is INCORRECT.
Reflection:
Error: I used the wrong answer format for a yes/no question.
Cause: I explained the comparison instead of submitting the required short answer.
Fix: Verify both entities, then finish with exactly yes or no.
"""


POLICY_REFLECTION_FEW_SHOT = """Example failed attempt:
Question: Which of two people is older?
Trajectory: I only checked one birth year and guessed the other.
Feedback: Answer is INCORRECT.
Reflection: Always verify both entities and compare the requested attribute directly before answering.

Example failed attempt:
Question: What role was an actor best known for?
Trajectory: I searched the show title, found a different actor, and finished with that actor's role.
Feedback: Answer is INCORRECT.
Reflection: Identify the target entity first, then retrieve evidence about that exact entity.

Example failed attempt:
Question: Were two people of the same nationality?
Trajectory: I found the nationalities but answered with an explanatory sentence.
Feedback: Answer is INCORRECT.
Reflection: For yes/no questions, submit exactly yes or no after checking both entities.
"""


COMPRESSED_REFLECTION_FEW_SHOT = """Example failed attempt:
Question: Which of two people is older?
Trajectory: I only checked one birth year and guessed the other.
Feedback: Answer is INCORRECT.
Reflection: Verify both entities before comparing.

Example failed attempt:
Question: What role was an actor best known for?
Trajectory: I searched the show title, found a different actor, and finished with that actor's role.
Feedback: Answer is INCORRECT.
Reflection: Identify actor before role lookup.

Example failed attempt:
Question: Were two people of the same nationality?
Trajectory: I answered with a full sentence instead of yes or no.
Feedback: Answer is INCORRECT.
Reflection: Yes/no questions need yes or no.
"""


REFLECTION_FEW_SHOTS = {
    "freeform": REFLECTION_FEW_SHOT,
    "structured": STRUCTURED_REFLECTION_FEW_SHOT,
    "policy": POLICY_REFLECTION_FEW_SHOT,
    "compressed": COMPRESSED_REFLECTION_FEW_SHOT,
}


def _format_memory(memory: list[str] | None) -> str:
    if not memory:
        return ""

    lines = ["Previous reflections to learn from:"]
    for i, reflection in enumerate(memory, 1):
        lines.append(f"[Reflection {i}]")
        lines.append(reflection.strip())
        lines.append("")

    lines.append(
        "Use these reflections only as advice. Do not continue the reflection format. "
        "Now answer the current question using the required output format."
    )
    lines.append("")

    return "\n".join(lines)


def build_cot_prompt(
    question: str,
    context: str = "",
    memory: list[str] | None = None,
    include_few_shot: bool = True,
) -> str:
    parts = []

    if include_few_shot:
        parts.append(COT_FEW_SHOT.strip() + "\n\n")

    parts.append(_format_memory(memory))

    if context:
        parts.append(f"Context:\n{context}\n\n")

    parts.append(f"Question: {question}\n\n")
    parts.append(
        "Reason briefly, then end with exactly one non-empty answer line:\n"
        "ANSWER: <short final answer>\n"
        "For yes/no questions, write exactly ANSWER: yes or ANSWER: no.\n"
        "Do not write Final answer:"
    )

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

    parts.append(
        "Next step.\n"
        "Respond with exactly two lines:\n"
        "Thought: <brief reasoning>\n"
        "Action: <Search[...] or Lookup[...] or Finish[...]>\n\n"
        "Important rules:\n"
        "- If the observations already contain enough evidence, use Finish[<short answer>] now.\n"
        "- If this is the last or near-last step, use Finish[<best short answer>] instead of searching again.\n"
        "- Never output Finish[].\n"
        "- For yes/no questions, use exactly Finish[yes] or Finish[no].\n"
        "- Do not write Final answer:.\n"
        "- Do not answer outside the Action field."
    )

    return "".join(parts)


def build_react_finalizer_prompt(
    question: str,
    trajectory: str,
    context: str = "",
) -> str:
    parts = []

    if context:
        parts.append(f"Available context:\n{context}\n\n")

    parts.append(f"Question: {question}\n\n")
    parts.append(f"ReAct trajectory:\n{trajectory}\n\n")
    parts.append(
        "The ReAct agent did not successfully finish. "
        "Based only on the question, context, and trajectory, provide the best short answer.\n\n"
        "Output exactly:\n"
        "ANSWER: <short final answer>\n\n"
        "For yes/no questions, output exactly ANSWER: yes or ANSWER: no."
    )

    return "".join(parts)

def build_reflector_prompt(
    question: str,
    trajectory: str,
    model_answer: str,
    feedback: str = "Answer is INCORRECT.",
    context: str = "",
    include_few_shot: bool = True,
    reflection_type: str = "freeform",
) -> str:
    parts = []

    if include_few_shot:
        few_shot = REFLECTION_FEW_SHOTS.get(reflection_type, REFLECTION_FEW_SHOT)
        parts.append(few_shot.strip() + "\n\n")

    parts.append("A QA attempt failed. Reflect on the failed attempt without using the gold answer.\n\n")
    parts.append(f"Question: {question}\n\n")

    if context:
        parts.append(f"Available context/retrieval source:\n{context}\n\n")

    parts.append(f"Failed trajectory or reasoning:\n{trajectory}\n\n")
    parts.append(f"Submitted answer: {model_answer}\n")
    parts.append(f"Feedback: {feedback}\n\n")

    if reflection_type == "structured":
        parts.append(
            "Write the reflection in exactly this format, with concise actionable content:\n"
            "Error: <what went wrong>\n"
            "Cause: <why it happened>\n"
            "Fix: <what to do differently next trial>\n\n"
            "Do not answer the question. Do not include an ANSWER line."
        )
    elif reflection_type == "policy":
        parts.append(
            "Extract one reusable reasoning policy that would help avoid similar mistakes "
            "in future tasks. Write one concise rule, not a task-specific answer.\n\n"
            "Do not answer the question. Do not include an ANSWER line."
        )
    elif reflection_type == "compressed":
        parts.append(
            "Write a compressed reflection in at most 12 words that captures the key lesson.\n\n"
            "Do not answer the question. Do not include an ANSWER line."
        )
    else:
        parts.append(
            "Write one concise first-person reflection with an actionable strategy for the next trial.\n\n"
            "Do not answer the question. Do not include an ANSWER line."
        )

    return "".join(parts)


# Backward-compatible aliases for your original code names.
ACTOR_SYSTEM_PROMPT = COT_SYSTEM_PROMPT


def build_actor_prompt(
    question: str,
    context: str = "",
    memory: list[str] | None = None,
) -> str:
    return build_cot_prompt(question=question, context=context, memory=memory)