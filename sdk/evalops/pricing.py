_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o":                       {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":                  {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":                {"input": 0.50,  "output": 1.50},
    "claude-3-5-sonnet-20241022":   {"input": 3.00,  "output": 15.00},
    "claude-3-haiku-20240307":      {"input": 0.25,  "output": 1.25},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD; returns 0.0 for unknown models."""
    prices = _PRICING_PER_1M.get(model)
    if prices is None:
        return 0.0
    return (
        input_tokens  * prices["input"]  / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )
