"""Agents that make no API calls.

They test the harness and give the floor lines on the scorecard:
- random: a seeded uniform guess. Its accuracy is about 1 / (number of options).
- first: always option A. It shows whether the shuffle hides the answer.
- refuse: always the refusal option. Its coverage must be 0.
- longest: the longest option text, without the refusal option. It is the
  "pick the longest option" shortcut from LAB-Bench issue #18. A seeded RNG
  breaks ties. A model that scores near this line may use the same shortcut.
- oracle: always correct. It reads the answer key, so it is a harness self-test
  only. The scorecard never publishes it.
"""

from __future__ import annotations

from ..mcq import MCQ, task_rng
from .base import AgentResult


class _Dummy:
    makes_api_calls = False
    name = "dummy"

    def config(self) -> dict:
        return {"agent": self.name, "kind": "dummy", "version": 1}


class RandomAgent(_Dummy):
    name = "random"

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        rng = task_rng(seed + 1, mcq.task_id)   # Not the shuffle RNG stream.
        return AgentResult(text=f"[ANSWER]{rng.choice(mcq.letters)}[/ANSWER]")


class FirstOptionAgent(_Dummy):
    name = "first"

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        return AgentResult(text="[ANSWER]A[/ANSWER]")


class RefuseAgent(_Dummy):
    name = "refuse"

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        return AgentResult(text=f"[ANSWER]{mcq.refuse_letter}[/ANSWER]")


class LongestOptionAgent(_Dummy):
    name = "longest"

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        cands = [(letter, opt) for letter, opt in zip(mcq.letters, mcq.options)
                 if letter != mcq.refuse_letter]
        top = max(len(opt) for _, opt in cands)
        tied = [letter for letter, opt in cands if len(opt) == top]
        rng = task_rng(seed + 2, mcq.task_id)   # Not the shuffle or random-agent stream.
        return AgentResult(text=f"[ANSWER]{rng.choice(tied)}[/ANSWER]")


class OracleAgent(_Dummy):
    name = "oracle"

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        return AgentResult(text=f"[ANSWER]{mcq.correct_letter}[/ANSWER]")


DUMMY_AGENTS = {a.name: a for a in (RandomAgent, FirstOptionAgent, RefuseAgent,
                                     LongestOptionAgent, OracleAgent)}
