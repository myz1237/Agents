import difflib
from pathlib import Path

from consts import SANDBOX_PATH
from utils import err, file_path_checker, get_relative_path, ok


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
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile="Old contents",
            tofile="New contents",
            lineterm="",
        )
        diff_text = "".join(diff) or "(Same contents)"
        return ok(f"File: {get_relative_path(safe_path, SANDBOX_PATH)} (Overwrite)\n\n{diff_text}")
    except Exception as e:
        return err(f"Error reading file: {e}")


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

    # unified_diff only show modified lines with some context, rather than all contents
    diff = difflib.unified_diff(
        # keepends => Keep \n at the end of a line
        old_str.splitlines(keepends=True),
        new_str.splitlines(keepends=True),
        fromfile="Old contents",
        tofile="New Contents",
        # add nothing at the end of a line
        lineterm="",
    )

    diff_text = "".join(diff) or "(Same contents)"
    scope = "All replace" if replace_all else "Unique replace"
    return ok(f"File: {get_relative_path(safe_path, SANDBOX_PATH)}\n: {scope}\n\n{diff_text}")
