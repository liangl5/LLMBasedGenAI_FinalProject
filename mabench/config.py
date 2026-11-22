"""Experiment configuration. Everything except `Flags` is shared by all systems."""
from dataclasses import dataclass

ALLOWED_PROJECT = "fresh-booster-489119-q5"
GCLOUD_CONFIG = "genai-final"
CREDIT_CONFIRMED_ENV = "MABENCH_CREDIT_CONFIRMED"


@dataclass(frozen=True)
class Flags:
    structured_handoff: bool = False
    shared_record: bool = False
    fix_loop_limit: bool = False


SYSTEMS = {
    "single": Flags(),
    "baseline": Flags(),
    "improved": Flags(True, True, True),
    "improved-no-handoff": Flags(False, True, True),
    "improved-no-record": Flags(True, False, True),
    "improved-no-limit": Flags(True, True, False),
}


@dataclass(frozen=True)
class RunConfig:
    system: str
    token_budget: int
    model: str
    max_steps: int = 80
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_fix_loops: int = 2
    compact_at_chars: int = 120_000

    def __post_init__(self):
        if self.system not in SYSTEMS:
            raise ValueError(f"unknown system {self.system!r}; choose from {sorted(SYSTEMS)}")
        if self.token_budget <= 0 or self.max_steps <= 0:
            raise ValueError("token_budget and max_steps must be positive")

    @property
    def flags(self) -> Flags:
        return SYSTEMS[self.system]
