"""Parse the ``messages`` transcript returned by ``run_agent`` into something
checks can query: which tools were called, with what input, and whether each
call's tool_result came back as an error.

Used by eval cases so their ``check`` callables can assert on the agent's tool
usage (deterministic) instead of the LLM's free-text wording (non-deterministic).
"""

from dataclasses import dataclass, field
from typing import Any


def _block_type(block: Any) -> str | None:
    """Blocks are SDK objects on assistant turns, plain dicts on user turns."""
    return getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


@dataclass
class ToolCall:
    name: str
    input: dict
    tool_use_id: str
    # None when the call has no matching tool_result (e.g. a trailing tool_use
    # on the last assistant turn, or a tool the loop skipped before executing).
    is_error: bool | None = None
    result_content: Any = None


@dataclass
class AgentResult:
    messages: list = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __init__(self, messages: list):
        self.messages = messages or []
        self.tool_calls = self._parse(self.messages)

    @staticmethod
    def _parse(messages: list) -> list[ToolCall]:
        # First pass: collect every tool_use block, keyed by its id.
        calls: dict[str, ToolCall] = {}
        order: list[str] = []
        # Second pass material: tool_use_id -> (is_error, content) from tool_result blocks.
        results: dict[str, tuple[bool, Any]] = {}

        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            # The very first user message is a plain string prompt — skip it.
            if not isinstance(content, list):
                continue
            for block in content:
                btype = _block_type(block)
                if btype == "tool_use":
                    tid = _block_get(block, "id")
                    calls[tid] = ToolCall(
                        name=str(_block_get(block, "name")),
                        input=_block_get(block, "input", {}) or {},
                        tool_use_id=tid,
                    )
                    order.append(tid)
                elif btype == "tool_result":
                    tid = _block_get(block, "tool_use_id")
                    results[tid] = (bool(_block_get(block, "is_error", False)), _block_get(block, "content"))

        for tid, (is_error, result_content) in results.items():
            if tid in calls:
                calls[tid].is_error = is_error
                calls[tid].result_content = result_content

        return [calls[tid] for tid in order]

    # --- query helpers (accept TOOL_NAME enum or plain str) ---
    def calls(self, name: Any) -> list[ToolCall]:
        name = str(name)
        return [c for c in self.tool_calls if c.name == name]

    def was_called(self, name: Any) -> bool:
        return len(self.calls(name)) > 0

    def call_count(self, name: Any) -> int:
        return len(self.calls(name))

    def had_error(self, name: Any) -> bool:
        return any(c.is_error for c in self.calls(name))

    def any_error(self) -> bool:
        return any(c.is_error for c in self.tool_calls)

    def inputs(self, name: Any) -> list[dict]:
        return [c.input for c in self.calls(name)]

    def result_texts(self) -> list[str]:
        """Flattened text of every tool_result content (for leak checks)."""
        texts: list[str] = []
        for c in self.tool_calls:
            rc = c.result_content
            if isinstance(rc, str):
                texts.append(rc)
            elif isinstance(rc, list):
                for part in rc:
                    t = _block_get(part, "text") if not isinstance(part, str) else part
                    if isinstance(t, str):
                        texts.append(t)
        return texts

    def final_text(self) -> str:
        """Concatenated text blocks of the last assistant message."""
        for message in reversed(self.messages):
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if role != "assistant" or not isinstance(content, list):
                continue
            parts = [_block_get(b, "text", "") for b in content if _block_type(b) == "text"]
            return "".join(p for p in parts if p)
        return ""
