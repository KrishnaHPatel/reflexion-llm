"""
Unified CoT / Reflexion / ReAct / ReAct+Reflexion agent for QA.

Supported modes:
- cot: retries CoT without reflection memory
- reflexion: CoT with verbal reflection memory after failed trials
- react: ReAct Search/Lookup/Finish loop without reflection memory
- react-reflexion: ReAct plus reflection memory after failed trials
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from evaluator import AnswerEvaluation, evaluate_answer, extract_answer
from llm import call_llm
from prompts import (
    COT_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    build_cot_prompt,
    build_react_prompt,
    build_reflector_prompt,
)

from react_env import ContextSearchEnvironment


AgentMode = Literal["cot", "reflexion", "react", "react-reflexion"]
ReflectionType = Literal["freeform", "structured", "policy", "compressed"]


@dataclass
class TrialResult:
    """Result of a single trial attempt."""

    trial_number: int
    reasoning: str
    model_answer: str
    is_correct: bool
    evaluation: AnswerEvaluation
    reflection: str = ""
    trajectory: list[dict] = field(default_factory=list)


@dataclass
class ReflexionResult:
    """Complete result for one question."""

    question: str
    gold_answer: str
    context: str
    mode: str
    reflection_type: str
    trials: list[TrialResult] = field(default_factory=list)
    final_answer: str = ""
    solved: bool = False
    memory: list[str] = field(default_factory=list)


class ReflexionAgent:
    """QA agent supporting CoT, ReAct, and optional Reflexion memory."""

    VALID_MODES = {"cot", "reflexion", "react", "react-reflexion"}
    VALID_REFLECTION_TYPES = {"freeform", "structured", "policy", "compressed"}

    def __init__(
        self,
        model: str | None = None,
        mode: AgentMode = "reflexion",
        reflection_type: ReflectionType = "freeform",
        max_trials: int = 3,
        max_memory: int = 3,
        max_react_steps: int = 6,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        include_few_shot: bool = True,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}. Choose one of {sorted(self.VALID_MODES)}")
        if reflection_type not in self.VALID_REFLECTION_TYPES:
            raise ValueError(
                f"Unsupported reflection_type {reflection_type!r}. "
                f"Choose one of {sorted(self.VALID_REFLECTION_TYPES)}"
            )

        self.model = model
        self.mode = mode
        self.reflection_type = reflection_type
        self.max_trials = max_trials
        self.max_memory = max_memory
        self.max_react_steps = max_react_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_few_shot = include_few_shot

    @property
    def uses_react(self) -> bool:
        return self.mode in {"react", "react-reflexion"}

    @property
    def uses_reflexion(self) -> bool:
        return self.mode in {"reflexion", "react-reflexion"}

    @staticmethod
    def _extract_action(response: str) -> str:
        """Extract the action portion from an LLM response."""
        action_line = re.search(r"Action\s*:\s*(.+)", response, flags=re.IGNORECASE | re.DOTALL)
        if action_line:
            return action_line.group(1).strip()

        bracketed = re.search(r"\b(Search|Lookup|Finish)\s*\[.*?\]", response, flags=re.IGNORECASE | re.DOTALL)
        if bracketed:
            return bracketed.group(0).strip()

        return response.strip()

    def _generate_cot_answer(self, question: str, context: str, memory: list[str]) -> tuple[str, str, list[dict]]:
        prompt = build_cot_prompt(
            question=question,
            context=context,
            memory=memory,
            include_few_shot=self.include_few_shot,
        )
        response = call_llm(
            prompt=prompt,
            system_prompt=COT_SYSTEM_PROMPT,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response, extract_answer(response), []

    def _generate_react_answer(self, question: str, context: str, memory: list[str]) -> tuple[str, str, list[dict]]:
        env = ContextSearchEnvironment(context=context)
        trajectory_text = ""
        structured_steps: list[dict] = []
        final_answer = ""

        for step_num in range(1, self.max_react_steps + 1):
            prompt = build_react_prompt(
                question=question,
                trajectory=trajectory_text,
                memory=memory,
                include_few_shot=self.include_few_shot,
            )
            response = call_llm(
                prompt=prompt,
                system_prompt=REACT_SYSTEM_PROMPT,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            action_text = self._extract_action(response)
            env_step = env.step(action_text)

            step_record = {
                "step_number": step_num,
                "llm_response": response,
                "action": action_text,
                "action_type": env_step.action_type,
                "argument": env_step.argument,
                "observation": env_step.observation,
            }
            structured_steps.append(step_record)

            trajectory_text += response.strip() + "\n"
            trajectory_text += f"Observation: {env_step.observation}\n"

            if env_step.finished:
                final_answer = env_step.answer
                break

        if not final_answer:
            # In ReAct, only Finish[...] submits an answer. If the model never
            # finishes, leave the answer empty instead of scoring an observation.
            final_answer = ""

        return trajectory_text, final_answer, structured_steps

    def _generate_reflection(
        self,
        question: str,
        context: str,
        trajectory: str,
        model_answer: str,
    ) -> str:
        prompt = build_reflector_prompt(
            question=question,
            context=context,
            trajectory=trajectory,
            model_answer=model_answer,
            feedback="Answer is INCORRECT.",
            include_few_shot=self.include_few_shot,
            reflection_type=self.reflection_type,
        )
        return call_llm(
            prompt=prompt,
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def run(self, question: str, gold_answer: str, context: str = "") -> ReflexionResult:
        result = ReflexionResult(
            question=question,
            gold_answer=gold_answer,
            context=context,
            mode=self.mode,
            reflection_type=self.reflection_type,
        )
        memory: list[str] = []

        for trial_num in range(1, self.max_trials + 1):
            if self.uses_react:
                reasoning, model_answer, trajectory = self._generate_react_answer(question, context, memory)
            else:
                reasoning, model_answer, trajectory = self._generate_cot_answer(question, context, memory)

            evaluation = evaluate_answer(model_answer, gold_answer)
            trial = TrialResult(
                trial_number=trial_num,
                reasoning=reasoning,
                model_answer=model_answer,
                is_correct=evaluation.exact_match,
                evaluation=evaluation,
                trajectory=trajectory,
            )

            result.trials.append(trial)
            result.final_answer = model_answer

            if evaluation.exact_match:
                result.solved = True
                result.memory = memory.copy()
                return result

            if self.uses_reflexion and trial_num < self.max_trials:
                reflection = self._generate_reflection(
                    question=question,
                    context=context,
                    trajectory=reasoning,
                    model_answer=model_answer,
                )
                trial.reflection = reflection
                memory.append(reflection)
                if len(memory) > self.max_memory:
                    memory = memory[-self.max_memory:]

        result.memory = memory.copy()
        return result
