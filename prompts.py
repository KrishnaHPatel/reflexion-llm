"""
Prompt Templates for Reflexion

This module contains all prompt templates used by the Actor (for Chain-of-Thought
reasoning) and the Reflector (for generating self-reflections on failures).
"""

# System prompt for the Actor: instructs the model to use step-by-step reasoning
ACTOR_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using step-by-step reasoning.
Think through the problem carefully before providing your final answer.
Always end your response with your final answer on a new line in the format:
ANSWER: <your answer>"""


def build_actor_prompt(
    question: str,
    context: str = "",
    memory: list[str] = None,
) -> str:
    """
    Build the prompt for the Actor to generate an answer.
    
    The Actor uses Chain-of-Thought style reasoning. If there are past reflections
    in memory, they are included to help avoid previous mistakes.
    
    Args:
        question: The question to answer
        context: Optional context/passage to help answer the question
        memory: List of past reflections from failed attempts
    
    Returns:
        The formatted prompt string
    """
    prompt_parts = []
    
    # Include past reflections if available
    if memory and len(memory) > 0:
        prompt_parts.append("Here are reflections from your previous attempts:\n")
        for i, reflection in enumerate(memory, 1):
            prompt_parts.append(f"Reflection {i}: {reflection}\n")
        prompt_parts.append("\nUse these reflections to avoid repeating mistakes.\n\n")
    
    # Include context if provided
    if context:
        prompt_parts.append(f"Context:\n{context}\n\n")
    
    # Add the question
    prompt_parts.append(f"Question: {question}\n\n")
    prompt_parts.append("Think step by step, then provide your final answer.")
    
    return "".join(prompt_parts)


# System prompt for the Reflector: instructs the model to analyze failures
REFLECTOR_SYSTEM_PROMPT = """You are a thoughtful assistant that helps improve reasoning by reflecting on mistakes.
When given a failed attempt at answering a question, analyze what went wrong and provide 
specific, actionable advice for how to answer correctly next time.
Be concise but specific. Focus on the reasoning error, not just that the answer was wrong."""


def build_reflector_prompt(
    question: str,
    context: str,
    model_answer: str,
    correct_answer: str,
    reasoning: str,
) -> str:
    """
    Build the prompt for the Reflector to generate a self-reflection.
    
    The Reflector analyzes a failed attempt and generates natural language
    feedback explaining what went wrong and how to improve.
    
    Args:
        question: The original question
        context: The context provided (if any)
        model_answer: The incorrect answer the model gave
        correct_answer: The correct answer
        reasoning: The model's reasoning trace from the failed attempt
    
    Returns:
        The formatted prompt string
    """
    prompt_parts = [
        "A previous attempt to answer this question was incorrect.\n\n",
        f"Question: {question}\n\n",
    ]
    
    if context:
        prompt_parts.append(f"Context: {context}\n\n")
    
    prompt_parts.extend([
        f"Previous reasoning:\n{reasoning}\n\n",
        f"Previous answer: {model_answer}\n",
        f"Correct answer: {correct_answer}\n\n",
        "Reflect on why the previous attempt was wrong. ",
        "What was the key mistake in the reasoning? ",
        "What should be done differently next time to get the correct answer? ",
        "Be specific and concise.",
    ])
    
    return "".join(prompt_parts)
