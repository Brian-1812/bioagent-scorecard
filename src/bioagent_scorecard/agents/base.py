"""The agent interface.

An adapter wraps one agent (a plain model, Biomni, a ToolUniverse agent, a
coding agent). The harness gives the adapter one MCQ and gets one AgentResult.
The adapter reports its own token use and tool calls. The harness measures the
time itself and computes the cost from the token counts and a price table.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from ..mcq import MCQ


@dataclasses.dataclass
class ToolCall:
    name: str
    args_summary: str = ""
    url: str = ""          # Any URL the tool fetched. The leak check reads it.
    ok: bool = True
    error: str = ""


@dataclasses.dataclass
class AgentResult:
    text: str | None                    # The final answer text from the agent.
    # Token contract: input_tokens is ALL prompt tokens, cached ones included.
    # cached_input_tokens is the part of input_tokens read from the cache.
    # Anthropic reports cache reads apart from input_tokens, so an adapter for
    # it must add them. pricing.cost_usd relies on this contract.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    #: The part of input_tokens written to the prompt cache. Anthropic bills
    #: cache writes above the normal input rate (1-hour writes: 2x input).
    cache_write_tokens: int = 0
    model: str = ""                     # The key into the price table.
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    trace: list[dict] = dataclasses.field(default_factory=list)
    error: str = ""                     # Set when the agent failed (crash, API error).
    safety_refusal: str = ""            # Set to the provider's refusal category, if any.
    #: The last usage-limit state the provider reported, for example
    #: {"five_hour": {"utilization": 0.05, "resets_at": 1795268400, "status": "allowed"}}.
    #: The runner uses it to stop a chunk before it uses too much of the plan.
    rate_limit: dict = dataclasses.field(default_factory=dict)
    #: True when the call failed because the plan's usage limit is reached.
    #: The runner then stops the run and does not score this task.
    usage_limit_hit: bool = False


class Agent(Protocol):
    name: str
    #: True when the agent calls a model API or another paid service.
    #: The dry-run mode accepts only agents with makes_api_calls = False.
    makes_api_calls: bool

    def config(self) -> dict:
        """Everything that changes the agent's answers: model, version, tools, prompt."""
        ...

    def answer(self, mcq: MCQ, seed: int) -> AgentResult:
        ...
