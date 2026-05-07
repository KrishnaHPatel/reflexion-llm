"""
A small retrieval environment for ReAct-style QA.

This intentionally avoids external dependencies. It supports the same action
shape used in the ReAct paper for HotpotQA-style tasks:

    Search[entity or query]
    Lookup[string]
    Finish[answer]

Instead of calling live Wikipedia, this environment searches the `context` field
provided in each JSONL example. That makes it deterministic and usable offline.
For a larger project, replace ContextSearchEnvironment with a Wikipedia or local
corpus retriever while keeping the same `step(action)` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "was", "were", "be", "been", "by", "as", "at", "from", "that",
    "this", "which", "who", "what", "when", "where", "why", "how",
}


@dataclass
class EnvironmentStep:
    action_type: str
    argument: str
    observation: str
    finished: bool = False
    answer: str = ""


@dataclass
class ContextSearchEnvironment:
    """Search/lookup environment over a single item's context text."""

    context: str = ""
    top_k: int = 3
    last_results: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sentences = self._split_sentences(self.context)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        # Split on sentence boundaries while preserving readable snippets.
        pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [p.strip() for p in pieces if p.strip()]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {t for t in tokens if t not in STOPWORDS}

    def _rank(self, query: str, candidates: Optional[list[str]] = None) -> list[str]:
        candidates = candidates if candidates is not None else self.sentences
        q_tokens = self._tokens(query)
        if not candidates:
            return []
        if not q_tokens:
            return candidates[: self.top_k]

        scored = []
        for sent in candidates:
            s_tokens = self._tokens(sent)
            overlap = len(q_tokens & s_tokens)
            phrase_bonus = 1 if query.lower() in sent.lower() else 0
            score = overlap + phrase_bonus
            if score > 0:
                scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sent for _, sent in scored[: self.top_k]]

    def search(self, query: str) -> str:
        results = self._rank(query)
        self.last_results = results
        if not results:
            return f"Could not find [{query}] in the provided context."
        return " ".join(results)

    def lookup(self, query: str) -> str:
        # Prefer looking within previous search results, then fall back to full context.
        base = self.last_results or self.sentences
        results = self._rank(query, base)
        if not results and self.last_results:
            results = self._rank(query, self.sentences)
        if not results:
            return f"No more results for [{query}] in the provided context."
        return " ".join(results)

    @staticmethod
    def parse_action(action_text: str) -> tuple[str, str] | None:
        """Parse Action lines like Search[x], Lookup[x], Finish[x]."""
        if not action_text:
            return None

        # Prefer explicit bracketed action.
        match = re.search(
            r"\b(Search|Lookup|Finish)\s*\[(.*?)\]",
            action_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).capitalize(), match.group(2).strip()

        # Also support `Action: Search query` as a fallback.
        match = re.search(r"\bAction\s*:\s*(Search|Lookup|Finish)\s*:?\s*(.+)", action_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).capitalize(), match.group(2).strip()

        return None

    def step(self, action_text: str) -> EnvironmentStep:
        parsed = self.parse_action(action_text)
        if parsed is None:
            return EnvironmentStep(
                action_type="Invalid",
                argument=action_text.strip(),
                observation="Invalid action. Use Search[...], Lookup[...], or Finish[...].",
            )

        action_type, argument = parsed
        if action_type == "Search":
            return EnvironmentStep(action_type, argument, self.search(argument))
        if action_type == "Lookup":
            return EnvironmentStep(action_type, argument, self.lookup(argument))
        if action_type == "Finish":
            return EnvironmentStep(action_type, argument, "Episode finished.", finished=True, answer=argument)

        return EnvironmentStep(action_type, argument, "Unknown action.")
