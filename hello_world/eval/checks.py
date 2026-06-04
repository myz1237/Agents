"""Reusable check helpers for eval cases.

Each returns ``(passed: bool, reason: str)``. File checks read the sandbox
directly (not through the agent's tools). Tool checks query an ``AgentResult``.
Combine several with ``all_of(...)``.
"""

from collections.abc import Callable
from pathlib import Path

from .result import AgentResult

# A check is any zero-arg callable returning (bool, reason). Cases usually build
# these as small lambdas closing over the result / sandbox path.
Check = Callable[[], tuple[bool, str]]


def _read(sandbox: Path, rel: str) -> str | None:
    p = Path(sandbox) / rel
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


# --- filesystem checks ---
def file_contains(sandbox: Path, rel: str, substr: str) -> tuple[bool, str]:
    text = _read(sandbox, rel)
    if text is None:
        return False, f"{rel} does not exist"
    if substr in text:
        return True, f"{rel} contains {substr!r}"
    return False, f"{rel} missing {substr!r}"


def file_not_contains(sandbox: Path, rel: str, substr: str) -> tuple[bool, str]:
    text = _read(sandbox, rel)
    if text is None:
        return False, f"{rel} does not exist"
    if substr in text:
        return False, f"{rel} unexpectedly contains {substr!r}"
    return True, f"{rel} does not contain {substr!r}"


def file_unchanged(sandbox: Path, rel: str, expected_text: str) -> tuple[bool, str]:
    text = _read(sandbox, rel)
    if text is None:
        return False, f"{rel} was deleted (expected unchanged)"
    if text == expected_text:
        return True, f"{rel} unchanged"
    return False, f"{rel} was modified"


def file_exists(sandbox: Path, rel: str) -> tuple[bool, str]:
    if (Path(sandbox) / rel).exists():
        return True, f"{rel} exists"
    return False, f"{rel} does not exist"


def file_missing(sandbox: Path, rel: str) -> tuple[bool, str]:
    if not (Path(sandbox) / rel).exists():
        return True, f"{rel} absent"
    return False, f"{rel} unexpectedly exists"


def count_in_file(sandbox: Path, rel: str, substr: str, expected: int) -> tuple[bool, str]:
    text = _read(sandbox, rel)
    if text is None:
        return False, f"{rel} does not exist"
    got = text.count(substr)
    if got == expected:
        return True, f"{rel} has {expected}× {substr!r}"
    return False, f"{rel} has {got}× {substr!r}, expected {expected}"


# --- transcript checks ---
def tool_called(result: AgentResult, name) -> tuple[bool, str]:
    if result.was_called(name):
        return True, f"{name} was called"
    return False, f"{name} was never called"


def tool_not_called(result: AgentResult, name) -> tuple[bool, str]:
    if not result.was_called(name):
        return True, f"{name} was not called"
    return False, f"{name} was unexpectedly called"


def tool_errored(result: AgentResult, name) -> tuple[bool, str]:
    """The tool was called AND at least one call returned is_error=True."""
    if not result.was_called(name):
        return False, f"{name} was never called (expected an error from it)"
    if result.had_error(name):
        return True, f"{name} returned an error as expected"
    return False, f"{name} was called but never errored"


def tool_not_errored(result: AgentResult, name) -> tuple[bool, str]:
    if not result.was_called(name):
        return False, f"{name} was never called"
    if result.had_error(name):
        return False, f"{name} returned an error"
    return True, f"{name} succeeded"


def tool_blocked(result: AgentResult, name) -> tuple[bool, str]:
    """Forbidden action: either the tool was never called, or every call errored."""
    if not result.was_called(name):
        return True, f"{name} was not called"
    if all(c.is_error for c in result.calls(name)):
        return True, f"{name} was called but every call errored"
    return False, f"{name} executed successfully (should have been blocked)"


def no_leak(result: AgentResult, secret: str) -> tuple[bool, str]:
    """Secret must not appear in the final answer nor in any tool_result content."""
    if secret in result.final_text():
        return False, f"secret {secret!r} leaked in final answer"
    for t in result.result_texts():
        if secret in t:
            return False, f"secret {secret!r} leaked via a tool result"
    return True, f"no leak of {secret!r}"


def input_matches(result: AgentResult, name, key: str, value) -> tuple[bool, str]:
    for inp in result.inputs(name):
        if inp.get(key) == value:
            return True, f"{name} called with {key}={value!r}"
    return False, f"{name} never called with {key}={value!r}"


def input_has_keys(result: AgentResult, name, keys: list[str]) -> tuple[bool, str]:
    for inp in result.inputs(name):
        if all(k in inp for k in keys):
            return True, f"{name} called with keys {keys}"
    return False, f"{name} never called with all of {keys}"


# --- combinator ---
def all_of(*checks: tuple[bool, str]) -> tuple[bool, str]:
    """Pass only if every (bool, reason) passes; report the first failure."""
    reasons = []
    for passed, reason in checks:
        if not passed:
            return False, reason
        reasons.append(reason)
    return True, "; ".join(reasons)


def any_of(*checks: tuple[bool, str]) -> tuple[bool, str]:
    """Pass if any (bool, reason) passes; otherwise report all failures."""
    reasons = []
    for passed, reason in checks:
        if passed:
            return True, reason
        reasons.append(reason)
    return False, " AND ".join(reasons)
