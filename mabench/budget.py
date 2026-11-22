"""Per-run token ledger and the cross-run dollar guard."""
import json
import threading
from collections import defaultdict
from pathlib import Path

MIN_OUTPUT_TOKENS = 64


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class SpendCapReached(Exception):
    """Stops the whole experiment, unlike BudgetExceeded which ends one run."""


class Ledger:
    def __init__(self, limit: int, max_steps: int, on_charge=None):
        self.limit, self.max_steps, self.on_charge = limit, max_steps, on_charge
        self.used = self.steps = self.input_tokens = self.output_tokens = 0
        self.by_role = defaultdict(int)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self, prompt_chars: int) -> int:
        """Return the output tokens this call may use, or stop the run before paying for it."""
        if self.steps >= self.max_steps:
            raise BudgetExceeded("max_steps")
        allowed = self.remaining - prompt_chars // 4  # shortcut: chars/4 estimate, use countTokens if overshoot is material
        if allowed < MIN_OUTPUT_TOKENS:
            raise BudgetExceeded("token_budget")
        return allowed

    def charge(self, role: str, input_tokens: int, output_tokens: int) -> None:
        self.steps += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.used += input_tokens + output_tokens
        self.by_role[role] += input_tokens + output_tokens
        if self.on_charge:
            self.on_charge(input_tokens, output_tokens)


class SpendGuard:
    """Durable dollar cap: every LLM call is appended, so restarts keep counting past spend."""

    def __init__(self, path, cap_usd: float, usd_per_m_in: float, usd_per_m_out: float):
        self.path, self.cap = Path(path), cap_usd
        self.rate_in, self.rate_out = usd_per_m_in, usd_per_m_out
        self.lock = threading.Lock()
        self.pending = 0.0
        self.total = 0.0
        if self.path.exists():
            self.total = sum(json.loads(l)["usd"] for l in self.path.read_text().splitlines() if l.strip())

    def _worst_case(self, tokens: int) -> float:
        return tokens / 1e6 * max(self.rate_in, self.rate_out)

    def admit(self, tokens: int) -> bool:
        cost = self._worst_case(tokens)
        with self.lock:
            if self.total + self.pending + cost > self.cap:
                return False
            self.pending += cost
            return True

    def release(self, tokens: int) -> None:
        with self.lock:
            self.pending = max(0.0, self.pending - self._worst_case(tokens))

    def add(self, input_tokens: int, output_tokens: int) -> None:
        usd = input_tokens / 1e6 * self.rate_in + output_tokens / 1e6 * self.rate_out
        with self.lock:
            self.total += usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"input": input_tokens, "output": output_tokens, "usd": usd}) + "\n")
            if self.total > self.cap:
                raise SpendCapReached(f"spent ${self.total:.2f} of ${self.cap:.2f}")
