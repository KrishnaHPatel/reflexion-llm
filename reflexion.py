"""
Reflexion Agent

This module implements the core Reflexion loop for reasoning tasks.
The agent attempts to answer questions using Chain-of-Thought reasoning,
evaluates its answers, and generates reflections on failures to improve
performance on subsequent trials.

The key insight from the Reflexion paper is that storing natural language
reflections in memory allows the model to learn from mistakes without
any weight updates - a form of "verbal reinforcement learning."
"""

from dataclasses import dataclass, field

from llm import call_llm
from evaluator import evaluate, extract_answer
from prompts import (
    ACTOR_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    build_actor_prompt,
    build_reflector_prompt,
)


@dataclass
class TrialResult:
    """Result of a single trial attempt."""
    trial_number: int
    reasoning: str
    model_answer: str
    is_correct: bool
    reflection: str = ""


@dataclass
class ReflexionResult:
    """Complete result of running Reflexion on one question."""
    question: str
    gold_answer: str
    context: str
    trials: list[TrialResult] = field(default_factory=list)
    final_answer: str = ""
    solved: bool = False
    memory: list[str] = field(default_factory=list)


class ReflexionAgent:
    """
    Reflexion agent for reasoning tasks.
    
    Implements the Actor-Evaluator-Reflector loop from the Reflexion paper:
    1. Actor generates an answer using CoT reasoning + memory
    2. Evaluator checks if the answer is correct
    3. If wrong, Reflector generates a reflection for memory
    4. Repeat until correct or max trials reached
    """
    
    def __init__(
        self,
        model: str = None,
        max_trials: int = 3,
        max_memory: int = 3,
        temperature: float = 0.7,
    ):
        """
        Initialize the Reflexion agent.
        
        Args:
            model: Ollama model name, such as "llama3.2" or "mistral"
            max_trials: Maximum attempts per question (default: 3)
            max_memory: Maximum reflections to store (default: 3)
            temperature: Sampling temperature for LLM calls
        """
        self.model = model
        self.max_trials = max_trials
        self.max_memory = max_memory
        self.temperature = temperature
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        memory: list[str],
    ) -> tuple[str, str]:
        """
        Actor: Generate an answer using Chain-of-Thought reasoning.
        
        Args:
            question: The question to answer
            context: Optional context passage
            memory: List of past reflections
        
        Returns:
            Tuple of (full_reasoning, extracted_answer)
        """
        prompt = build_actor_prompt(question, context, memory)
        
        response = call_llm(
            prompt=prompt,
            system_prompt=ACTOR_SYSTEM_PROMPT,
            model=self.model,
            temperature=self.temperature,
        )
        
        answer = extract_answer(response)
        return response, answer
    
    def _generate_reflection(
        self,
        question: str,
        context: str,
        reasoning: str,
        model_answer: str,
        gold_answer: str,
    ) -> str:
        """
        Reflector: Generate a reflection explaining the mistake.
        
        This is the key component that enables learning. The reflection
        provides actionable feedback that helps the model avoid similar
        mistakes in future trials.
        
        Args:
            question: The original question
            context: The context passage
            reasoning: The model's reasoning from the failed attempt
            model_answer: The incorrect answer given
            gold_answer: The correct answer
        
        Returns:
            Natural language reflection on the mistake
        """
        prompt = build_reflector_prompt(
            question=question,
            context=context,
            model_answer=model_answer,
            correct_answer=gold_answer,
            reasoning=reasoning,
        )
        
        reflection = call_llm(
            prompt=prompt,
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            model=self.model,
            temperature=self.temperature,
        )
        
        return reflection
    
    def run(self, question: str, gold_answer: str, context: str = "") -> ReflexionResult:
        """
        Run the Reflexion loop on a single question.
        
        Attempts to answer the question up to max_trials times. After each
        failed attempt, generates a reflection and adds it to memory for
        the next trial.
        
        Args:
            question: The question to answer
            gold_answer: The correct answer for evaluation
            context: Optional context passage
        
        Returns:
            ReflexionResult containing all trial information
        """
        result = ReflexionResult(
            question=question,
            gold_answer=gold_answer,
            context=context,
        )
        
        memory: list[str] = []
        
        for trial_num in range(1, self.max_trials + 1):
            # Actor generates answer with current memory
            reasoning, model_answer = self._generate_answer(
                question=question,
                context=context,
                memory=memory,
            )
            
            # Evaluator checks correctness
            is_correct = evaluate(model_answer, gold_answer)
            
            trial = TrialResult(
                trial_number=trial_num,
                reasoning=reasoning,
                model_answer=model_answer,
                is_correct=is_correct,
            )
            
            if is_correct:
                # Success - no reflection needed
                result.trials.append(trial)
                result.final_answer = model_answer
                result.solved = True
                result.memory = memory.copy()
                return result
            
            # Failed - generate reflection for next trial
            if trial_num < self.max_trials:
                reflection = self._generate_reflection(
                    question=question,
                    context=context,
                    reasoning=reasoning,
                    model_answer=model_answer,
                    gold_answer=gold_answer,
                )
                trial.reflection = reflection
                
                # Add reflection to memory, keeping only the most recent
                memory.append(reflection)
                if len(memory) > self.max_memory:
                    memory = memory[-self.max_memory:]
            
            result.trials.append(trial)
        
        # All trials exhausted without success
        result.final_answer = result.trials[-1].model_answer
        result.memory = memory.copy()
        return result
