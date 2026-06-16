"""Write-tool previews for the LangGraph agent.

Rewritten from tools/preview.py. The `_render_diff` rendering is kept as-is.
The core difference is error handling: instead of returning {content, is_error},
each preview now **raises** a custom exception (from exceptions.py) on failure
and **returns the rendered diff string** on success.

The caller (the preview node in compile_interrupt.py) is responsible for catching
the exception and turning it into an error ToolMessage — same pattern as run_one_tool.

Path-safety helpers are reused from lang_graph.tools so the sandbox rules stay
identical to the real write tools.
"""

import difflib
from pathlib import Path

from consts import TOOL_NAME
from lang_graph.exceptions import InvalidArgumentError, StringNotFoundError
from lang_graph.tools import _existing_file, _read_text, _rel

# ANSI colors for a readable terminal diff
_RED = "\033[31m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _render_diff(old_text: str, new_text: str, fromfile: str, tofile: str, n: int = 3) -> str:
    """Build a colorized unified diff between two strings.

    Lines are split WITHOUT keepends and ``lineterm`` is empty, so every line
    difflib yields is newline-free; we re-join with "\\n". This keeps the header
    lines (---, +++, @@) and the body consistently separated, avoiding the
    run-together output you get when keepends and lineterm disagree.

    ``n`` is the number of unchanged context lines shown around each change.
    """
    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
        n=n,
    )

    colored: list[str] = []
    for line in diff:
        if line.startswith(("---", "+++")):
            colored.append(f"{_DIM}{line}{_RESET}")
        elif line.startswith("@@"):
            colored.append(f"{_CYAN}{line}{_RESET}")
        elif line.startswith("+"):
            colored.append(f"{_GREEN}{line}{_RESET}")
        elif line.startswith("-"):
            colored.append(f"{_RED}{line}{_RESET}")
        else:
            colored.append(line)

    return "\n".join(colored) or "(Same contents)"


def write_file_preview(args: dict) -> str:
    """Render a preview for a write_file_in_sandbox call.

    Returns the colorized diff string. Raises on bad input or unreadable file.
    """
    path: str | None = args.get("path")
    content: str = args.get("content", "")

    if path is None:
        raise InvalidArgumentError("path is required to preview a write.")
    if content == "":
        raise InvalidArgumentError("content is required to preview a write.")

    safe: Path = _existing_file(path)  # raises PathTraversal / FileNotFound / NotAFile
    old = _read_text(safe)  # raises FileDecodeError on binary
    diff_text = _render_diff(old, content, "Old contents", "New contents")
    return f"File: {_rel(safe)} (Overwrite)\n\n{diff_text}"


def str_replace_preview(args: dict) -> str:
    """Render a preview for a string_replace call.

    Diffs the WHOLE file before/after the replacement (not just the snippet),
    so the preview shows surrounding context and real line numbers.
    Returns the colorized diff string. Raises on bad input or missing old_str.
    """
    path: str | None = args.get("path")
    old_str: str = args.get("old_str", "")
    new_str: str = args.get("new_str", "")
    replace_all: bool = args.get("replace_all", False)

    if path is None:
        raise InvalidArgumentError("path is required to preview a replace.")
    if old_str == "":
        raise InvalidArgumentError("old_str is required to preview a replace.")
    if new_str == "":
        raise InvalidArgumentError("new_str is required to preview a replace.")

    safe: Path = _existing_file(path)
    old_file = _read_text(safe)
    rel = _rel(safe)

    if old_str not in old_file:
        raise StringNotFoundError(f"old_str not found in {rel}; cannot preview the change.")

    # count=-1 replaces every occurrence; count=1 replaces only the first.
    new_file = old_file.replace(old_str, new_str, -1 if replace_all else 1)

    diff_text = _render_diff(old_file, new_file, "Old contents", "New contents")
    scope = "All replace" if replace_all else "Unique replace"
    return f"File: {rel}: {scope}\n\n{diff_text}"


# Map tool name -> preview function (mirrors tools.WRITE_PREVIEW_MAP).
# StrEnum keys, so a plain string tool name (call["name"]) looks up fine.
LG_WRITE_PREVIEW_MAP = {
    TOOL_NAME.WRITE_FILE_IN_SANDBOX: write_file_preview,
    TOOL_NAME.STRING_REPLACE: str_replace_preview,
}
