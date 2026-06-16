"""Native LangChain tools (for LangGraph's ToolNode).

Unlike the tools/ package (which takes a dict and returns {content, is_error}),
each tool here:
  - uses typed parameters (LangChain derives the model-facing input schema from them);
  - returns a string on success;
  - raises a custom exception from exceptions.py on failure -- ToolNode catches it
    and feeds the exception message back to the model, so no manual error wrapping.

Security constraints (paths can't escape the sandbox, secret files are unreadable,
shell commands are whitelisted) reuse the constants from consts.py, matching the
original agent.
"""

import ast
import operator
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from consts import (
    ALLOWED_COMMANDS,
    SANDBOX_DIR,
    SANDBOX_PATH,
    SECRET_FILENAME_EXACT,
    SECRET_FILENAME_KEYWORDS,
    SECRET_FILENAME_PREFIXES,
    SECRET_FILENAME_SUFFIXES,
    TOOL_NAME,
)
from lang_graph.exceptions import (
    AmbiguousReplacementError,
    CalculationError,
    CommandNotAllowedError,
    CommandTimeoutError,
    EmptyContentError,
    FileDecodeError,
    FileNotFoundInSandboxError,
    InvalidArgumentError,
    InvalidTimezoneError,
    NotADirectoryInSandboxError,
    NotAFileError,
    PathTraversalError,
    SearchFailedError,
    SecretFileAccessError,
    StringNotFoundError,
)


# ---------------------------------------------------------------------------
# Internal helpers: path safety and validation. Always raise on failure.
# ---------------------------------------------------------------------------
def _safe_path(user_input_path: str) -> Path:
    """Resolve a relative/absolute path inside the sandbox; raise PathTraversalError if it escapes."""
    sandbox = Path(SANDBOX_DIR).resolve()
    target = sandbox.joinpath(user_input_path).resolve()
    if not target.is_relative_to(sandbox):
        raise PathTraversalError(
            f"Path {user_input_path!r} escapes the sandbox directory; refused. Use a path inside the sandbox."
        )
    return target


def _existing_file(user_input_path: str) -> Path:
    """Resolve and confirm the target is an existing file."""
    p = _safe_path(user_input_path)
    if not p.exists():
        raise FileNotFoundInSandboxError(
            f"File not found: {user_input_path}. Use {TOOL_NAME.LIST_DIRECTORY} to see what exists."
        )
    if not p.is_file():
        raise NotAFileError(f"{user_input_path} is not a file (it may be a directory).")
    return p


def _existing_dir(user_input_path: str) -> Path:
    """Resolve and confirm the target is an existing directory."""
    p = _safe_path(user_input_path)
    if not p.exists():
        raise FileNotFoundInSandboxError(f"Directory not found: {user_input_path}")
    if not p.is_dir():
        raise NotADirectoryInSandboxError(f"{user_input_path} is not a directory.")
    return p


def _is_secret_file(name: str) -> bool:
    """True if a filename looks like a secret/credential file (case-insensitive)."""
    n = name.lower()
    if n in SECRET_FILENAME_EXACT:
        return True
    if n.startswith(SECRET_FILENAME_PREFIXES):
        return True
    if n.endswith(SECRET_FILENAME_SUFFIXES):
        return True
    return any(kw in n for kw in SECRET_FILENAME_KEYWORDS)


def _read_text(path: Path) -> str:
    """Read a file as UTF-8; raise FileDecodeError if it isn't text."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise FileDecodeError(f"{path.name} is not valid UTF-8 text (looks binary); cannot read.") from e


def _rel(path: Path) -> str:
    """Path relative to the sandbox root, to hide the absolute path when echoing back."""
    return str(path.resolve().relative_to(SANDBOX_PATH.resolve()))


# ---------------------------------------------------------------------------
# Safe arithmetic evaluation (only numbers and + - * / // % ** plus parentheses)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalculationError(f"Unsupported constant: {node.value!r}")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise CalculationError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise CalculationError(f"Syntax error in expression: {e}") from e
    try:
        return _eval(tree.body)
    except (ZeroDivisionError, OverflowError) as e:
        raise CalculationError(f"Calculation error: {e}") from e


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@tool(TOOL_NAME.GET_CURRENT_TIME, parse_docstring=True)
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """Get the current time. Invoke it when user asks for the current time.

    Args:
        timezone: The timezone to get the current time for, as an IANA name like
            'Asia/Shanghai' or 'UTC'. Default is Asia/Shanghai (UTC+8).
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as e:
        raise InvalidTimezoneError(
            f"Unknown timezone: {timezone!r}. Use an IANA name like 'Asia/Tokyo' or 'UTC'."
        ) from e
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z")
    return f"The current time in {timezone} is {now}."


@tool(TOOL_NAME.TIME_OFFSET, parse_docstring=True)
def time_offset(base_time: str, offset_seconds: int) -> str:
    """Calculate a time offset. Invoke it when user asks to calculate a time offset.

    Args:
        base_time: The base time, in ISO format, e.g., '2026-05-29 23:30:24'.
        offset_seconds: The offset in seconds, positive for later, negative for earlier.
    """
    try:
        base_dt = datetime.fromisoformat(base_time)
    except ValueError as e:
        raise InvalidArgumentError(f"Invalid base_time format (ISO format required): {e}") from e
    return (base_dt + timedelta(seconds=offset_seconds)).isoformat()


@tool(TOOL_NAME.CALCULATE, parse_docstring=True)
def calculate(expression: str) -> str:
    """Execute pure numerical operations (addition, subtraction, multiplication,
    division, exponentiation, square root, etc.). Do not use for time calculations;
    use time_offset for that.

    Args:
        expression: The mathematical expression to calculate, like 2 * 5 + 1.
    """
    result = _safe_eval(expression)
    return f"The result of {expression} is {result}."


@tool(TOOL_NAME.READ_FILE_IN_SANDBOX, parse_docstring=True)
def read_file_in_sandbox(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_lines: int = 500,
) -> str:
    """Read the contents of a file. Invoke it when user asks to read a file.

    Args:
        path: The absolute or relative path to the file to read.
        start_line: First line to read (1-indexed, inclusive). Optional.
        end_line: Last line to read (1-indexed, inclusive). Optional, defaults to end of file.
        max_lines: The maximum number of lines to read from the file. Default is 500.
    """
    if _is_secret_file(Path(path).name):
        raise SecretFileAccessError(
            f"Refused: {Path(path).name!r} looks like a secret/credential file; "
            "its contents cannot be read or disclosed."
        )

    safe = _existing_file(path)
    lines = _read_text(safe).splitlines()
    total = len(lines)

    # No line range given: read normally, capped at max_lines.
    if start_line is None and end_line is None:
        return "\n".join(lines[:max_lines])

    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = total

    if start_line < 1:
        raise InvalidArgumentError(f"start_line must be >= 1, got {start_line}.")
    if start_line > total:
        raise InvalidArgumentError(f"start_line ({start_line}) is past the end of the file ({total} lines).")
    if end_line < start_line:
        raise InvalidArgumentError(f"end_line ({end_line}) must be >= start_line ({start_line}).")

    sliced = lines[start_line - 1 : end_line]
    lined = "\n".join(f"{n + start_line}: {line}" for n, line in enumerate(sliced))
    return f"Lines {start_line}-{end_line} of {total}:\n{lined}"


@tool(TOOL_NAME.LIST_DIRECTORY, parse_docstring=True)
def list_directory(dir_path: str) -> str:
    """List the contents of a directory. Invoke it when user asks to list a directory.

    Args:
        dir_path: The absolute or relative path to the directory to list.
    """
    safe = _existing_dir(dir_path)
    entries = [
        ("📁" if entry.is_dir() else "📄") + entry.name for entry in safe.iterdir() if not entry.name.startswith(".")
    ]
    return "\n".join(entries) if entries else "(empty directory)"


@tool(TOOL_NAME.WRITE_FILE_IN_SANDBOX, parse_docstring=True)
def write_file_in_sandbox(path: str, content: str) -> str:
    """Write content to a file. Invoke it when user asks to write to a file.

    Args:
        path: The absolute or relative path to the file to write.
        content: The content to write to the file.
    """
    if content.strip() == "":
        raise EmptyContentError("Content is empty; please provide content to write.")

    safe = _safe_path(path)  # writing allows a non-existent target; only validate path safety
    safe.parent.mkdir(parents=True, exist_ok=True)
    is_existing = safe.is_file()
    with safe.open("a", encoding="utf-8") as f:
        f.write(("\n" if is_existing else "") + content)
    return f"Successfully wrote to file: {path}"


@tool(TOOL_NAME.STRING_REPLACE, parse_docstring=True)
def string_replace(path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
    """Replace strings in a file. Invoke it when user asks to replace strings in a file.
    If you find multiple occurrences of the string to be replaced, ask users which one
    to replace, otherwise no action will be taken. Please make the old_str unique,
    multiple occurrences of the same old_str will be denied for safety. It's recommended
    to modify parts of contents of a file, more efficient than the tool combination of
    read_file_in_sandbox + write_file_in_sandbox. If you wanna append new content, new_str
    should be old_str plus the new content. Do not add all contents as old_str, only pick
    up what you wanna change.

    Args:
        path: The path to the file to replace strings in.
        old_str: The string to be replaced.
        new_str: The string to replace with.
        replace_all: Set true to replace all occurrences at once. When multiple
            occurrences exist and the user wants all of them changed, PREFER setting
            replace_all=true with a minimal old_str, rather than expanding old_str to
            cover a large block. Default false.
    """
    if old_str == "":
        raise InvalidArgumentError("old_str is empty; please provide the string to replace.")
    if new_str == "":
        raise InvalidArgumentError("new_str is empty; please provide the replacement string.")

    safe = _existing_file(path)
    content = _read_text(safe)
    occurrences = content.count(old_str)
    rel = _rel(safe)

    if occurrences == 0:
        preview = content[:50] + "..." if len(content) > 50 else content
        raise StringNotFoundError(
            f"Could not find the string to replace in {rel}. File start preview: {preview}. Please check and retry."
        )
    if occurrences > 1 and not replace_all:
        raise AmbiguousReplacementError(
            f"Found {occurrences} occurrences of old_str in {rel}. Make old_str unique, "
            "or set replace_all=True if you intend to replace all of them."
        )

    new_content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
    safe.write_text(new_content, encoding="utf-8")
    return f"Successfully replaced string in {rel}. File size: {len(new_content)} chars"


@tool(TOOL_NAME.RUN_LIMITED_SHELL_COMMAND, parse_docstring=True)
def run_limited_shell_command(command: str) -> str:
    """Run a limited shell command. Invoke it when user asks to run a shell command.
    Only a limited set of safe commands are allowed (ls, cat, grep, find, wc, head,
    tail, echo, pwd).

    Args:
        command: The shell command to run.
    """
    if not command.strip():
        raise InvalidArgumentError("Command is empty; please provide a shell command to run.")

    parts = command.split()
    if parts[0] not in ALLOWED_COMMANDS:
        raise CommandNotAllowedError(
            f"Command {parts[0]!r} is not allowed. Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}."
        )

    try:
        # shell=False + arg list: avoids shell injection (the original used shell=True, a hazard).
        result = subprocess.run(parts, cwd=SANDBOX_DIR, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired as e:
        raise CommandTimeoutError("Command exceeded 10 seconds and was timed out.") from e

    output = result.stdout.strip() if result.stdout else result.stderr.strip()
    if len(output) > 500:
        return f"Command output is too long to display: {output[:500]}..."
    return f"stdout: {output}"


@tool(TOOL_NAME.SEARCH_CODE, parse_docstring=True)
def search_code(pattern: str, file_pattern: str = "", max_results: int = 50) -> str:
    """Search for text or regex patterns across files in the sandbox using ripgrep.
    Returns matching file names, line numbers, and content. Much faster than reading
    files one by one. Use this for code exploration, finding function definitions,
    tracking usages, etc.

    Args:
        pattern: Text or regex to search, e.g. 'def reverse' or 'import\\s+\\w+'.
        file_pattern: Glob filter for filenames, e.g. '*.py' for Python files only.
            Empty = all files.
        max_results: Max number of matches to return. Default 50.
    """
    if pattern.strip() == "":
        raise InvalidArgumentError("Search pattern is empty; please provide text or a regex.")

    cmd = ["rg", "--max-count", "10", "--max-columns", "200", "--line-number", "--no-heading", "--color", "never"]
    if file_pattern:
        cmd.extend(["--glob", file_pattern])
    cmd.append(pattern)
    cmd.append(str(SANDBOX_PATH))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=SANDBOX_DIR)
    except subprocess.TimeoutExpired as e:
        raise SearchFailedError("Search timed out.") from e

    # rg: 0 = matches, 1 = no matches, 2 = error
    if result.returncode == 1:
        return "No match found"
    if result.returncode not in (0, 1):
        raise SearchFailedError(f"Search failed: {result.stderr.strip()}")

    stripped = result.stdout.strip()
    lines = stripped.split("\n")
    if len(lines) > max_results:
        lines = lines[:max_results]
        lines.append(f"...and more results (truncated to {max_results}).")
        stripped = "\n".join(lines)
    return stripped.replace(str(SANDBOX_PATH) + "/", "")  # strip the absolute-path prefix


LG_TOOL_MAP = {
    TOOL_NAME.GET_CURRENT_TIME: get_current_time,
    TOOL_NAME.CALCULATE: calculate,
    TOOL_NAME.TIME_OFFSET: time_offset,
    TOOL_NAME.READ_FILE_IN_SANDBOX: read_file_in_sandbox,
    TOOL_NAME.LIST_DIRECTORY: list_directory,
    TOOL_NAME.WRITE_FILE_IN_SANDBOX: write_file_in_sandbox,
    TOOL_NAME.RUN_LIMITED_SHELL_COMMAND: run_limited_shell_command,
    TOOL_NAME.STRING_REPLACE: string_replace,
    TOOL_NAME.SEARCH_CODE: search_code,
}

# Tool list exported for the graph to use
LG_TOOLS = [
    get_current_time,
    time_offset,
    calculate,
    read_file_in_sandbox,
    list_directory,
    write_file_in_sandbox,
    string_replace,
    run_limited_shell_command,
    search_code,
]
