"""Token prices and cost.

prices.json holds USD per million tokens per model, with the date and the
source page. The run manifest copies the price rows it used, so an old run
keeps its old prices after the table changes.
"""

from __future__ import annotations

import json
from importlib import resources


def load_prices() -> dict:
    return json.loads(resources.files(__package__).joinpath("prices.json").read_text("utf-8"))


def cost_usd(prices: dict, model: str, input_tokens: int, output_tokens: int,
             cached_input_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    """input_tokens includes cache reads and cache writes (AgentResult contract)."""
    if not model:
        return 0.0
    row = prices["models"].get(model)
    if row is None:
        raise KeyError(f"no price for model {model!r}; add it to prices.json")
    fresh = max(0, input_tokens - cached_input_tokens - cache_write_tokens)
    cached_rate = row.get("cached_input", row["input"])
    write_rate = row.get("cache_write", row["input"])
    return (fresh * row["input"] + cached_input_tokens * cached_rate
            + cache_write_tokens * write_rate + output_tokens * row["output"]) / 1_000_000
