import difflib
from pathlib import Path

from consts import SANDBOX_PATH
from utils import err, file_path_checker, get_relative_path, ok

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


def write_file_preview(tool_input: dict) -> dict:
    path: str | None = tool_input.get("path")
    content: str = tool_input.get("content", "")

    if path is None:
        return err("path is unavailable, please add it to the tool.")
    if content == "":
        return err("content is unavailable, please add it to the tool.")

    path_checker = file_path_checker(path)

    if path_checker["is_error"]:
        return path_checker

    safe_path: Path = path_checker["content"]

    # try to read with utf-8 encoding, if fails, return error message
    try:
        old = safe_path.read_text(encoding="utf-8")
    except Exception as e:
        return err(f"Error reading file: {e}")

    diff_text = _render_diff(old, content, "Old contents", "New contents")
    return ok(f"File: {get_relative_path(safe_path, SANDBOX_PATH)} (Overwrite)\n\n{diff_text}")


def str_replace_preview(tool_input: dict) -> dict:
    path: str | None = tool_input.get("path")
    old_str: str = tool_input.get("old_str", "")
    new_str: str = tool_input.get("new_str", "")
    replace_all: bool = tool_input.get("replace_all", False)

    if path is None:
        return err("path is unavailable, please add it to the tool.")
    if old_str == "":
        return err("old_str is empty, please add it into the tool.")
    if new_str == "":
        return err("new_str is empty, please add it into the tool.")

    path_checker = file_path_checker(path)

    if path_checker["is_error"]:
        return path_checker

    safe_path: Path = path_checker["content"]

    # Diff the WHOLE file before/after the replacement (not just the snippet),
    # so the preview shows surrounding context and real line numbers.
    try:
        old_file = safe_path.read_text(encoding="utf-8")
    except Exception as e:
        return err(f"Error reading file: {e}")

    relative_path = get_relative_path(safe_path, SANDBOX_PATH)
    if old_str not in old_file:
        return err(f"old_str not found in {relative_path}; cannot preview the change.")

    # count=-1 replaces every occurrence; count=1 replaces only the first.
    new_file = old_file.replace(old_str, new_str, -1 if replace_all else 1)

    diff_text = _render_diff(old_file, new_file, "Old contents", "New contents")
    scope = "All replace" if replace_all else "Unique replace"
    return ok(f"File: {relative_path}: {scope}\n\n{diff_text}")
