"""Agent adapters.

- dummy.py: agents with no API calls (random, first, refuse, longest, oracle).
- claude_cli.py: rows A and B through headless Claude Code on a Claude
  subscription. Row A, plain model with no tools: `claude-cli:<model>`. Row B, model
  plus one sandboxed Python tool: `claude-sandbox:<model>`.
"""

from .base import Agent, AgentResult, ToolCall
from .dummy import DUMMY_AGENTS

MODEL_AGENT_PREFIXES = ("claude-cli:", "claude-sandbox:")


def get_agent(name: str, **kw) -> Agent:
    if name in DUMMY_AGENTS:
        return DUMMY_AGENTS[name]()
    prefix, _, model = name.partition(":")
    if f"{prefix}:" in MODEL_AGENT_PREFIXES:
        from .claude_cli import ClaudeCLIAgent, ClaudeSandboxAgent
        if not model:
            raise KeyError(f"{prefix} needs a model, for example {prefix}:claude-haiku-5-5")
        cls = ClaudeSandboxAgent if prefix == "claude-sandbox" else ClaudeCLIAgent
        return cls(model, **kw)
    raise KeyError(f"unknown agent {name!r}; dummy agents: {sorted(DUMMY_AGENTS)}; "
                   f"model agents: claude-cli:<model>, claude-sandbox:<model>")


__all__ = ["Agent", "AgentResult", "ToolCall", "DUMMY_AGENTS", "MODEL_AGENT_PREFIXES", "get_agent"]
