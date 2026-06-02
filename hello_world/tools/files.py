from pathlib import Path

from consts import SANDBOX_DIR, SANDBOX_PATH, TOOL_NAME
from utils import err, ok


def get_relative_path(path: Path, relative_to: Path) -> Path:
    return path.resolve().relative_to(relative_to.resolve())


def get_previews(content: str, target_str: str) -> list:
    if not target_str:
        return []

    previews = []
    start = 0
    while True:
        index = content.find(target_str, start)
        if index == -1:
            break
        # 5 chars before + the match itself + 5 chars after.
        # max(0, ...) guards the start so a negative index doesn't slice from
        # the tail; the end can overshoot len(content) — slicing clamps it.
        begin = max(0, index - 5)
        end = index + len(target_str) + 5
        previews.append(content[begin:end])
        # Step past this match so occurrences are counted non-overlapping,
        # matching str.count()/content.count(target_str).
        start = index + len(target_str)
    return previews


def get_safe_path(user_input_path: str) -> Path:
    abs_sandbox_path = Path(SANDBOX_DIR).resolve()
    abs_target_path = abs_sandbox_path.joinpath(user_input_path).resolve()
    if not abs_target_path.is_relative_to(abs_sandbox_path):
        raise ValueError("Path traversal detected, please provide a path within the sandbox directory.")
    return abs_target_path


def file_path_checker(user_input_path: str) -> dict:
    try:
        safe_path = get_safe_path(user_input_path)
    except ValueError as e:
        return err(str(e))

    if not safe_path.exists():
        return err(f"File not found: {user_input_path}")
    if not safe_path.is_file():
        return err(f"Not a file: {user_input_path}")
    return ok(safe_path)


def pure_file_path_checker(user_input_path: str) -> dict:
    try:
        return ok(get_safe_path(user_input_path))
    except ValueError as e:
        return err(str(e))


def read_file_as_string(file_path: Path) -> dict:
    try:
        return ok(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return err(f"File not found: {file_path}. Use {TOOL_NAME.LIST_DIRECTORY} to see what exists.")
    except IsADirectoryError:
        return err(f"{file_path} is a directory, not a file. Use {TOOL_NAME.LIST_DIRECTORY} to inspect it.")
    except PermissionError:
        return err(f"Permission denied for {file_path}. It exists but isn't readable. Please ask for permission.")
    except UnicodeDecodeError:
        return err(f"{file_path} is not valid UTF-8 text (looks binary). Cannot read as a string.")
    except OSError as e:
        # Catch-all for other I/O errors. MUST be last — the cases above are
        # all subclasses of OSError.
        return err(f"Could not read {file_path}: {e}")


def folder_path_checker(user_input_path: str) -> dict:
    try:
        safe_path = get_safe_path(user_input_path)
    except ValueError as e:
        return err(str(e))

    if not safe_path.exists():
        return err(f"Directory not found: {user_input_path}")
    if not safe_path.is_dir():
        return err(f"Not a directory: {user_input_path}")
    return ok(safe_path)


def read_file_in_sandbox(tool_input: dict) -> dict:
    file_path: str = tool_input.get("file_path")
    max_lines: int = tool_input.get("max_lines", 500)

    file_check_result = file_path_checker(file_path)
    if file_check_result["is_error"]:
        return file_check_result
    safe_path: Path = file_check_result["content"]

    # try to read with utf-8 encoding, if fails, return error message
    try:
        with safe_path.open("r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return ok("\n".join(lines))
    except Exception as e:
        return err(f"Error reading file: {e}")


def list_directory(tool_input: dict) -> dict:
    dir_path: str = tool_input.get("dir_path")

    folder_check_result = folder_path_checker(dir_path)
    if folder_check_result["is_error"]:
        return folder_check_result
    safe_path: Path = folder_check_result["content"]

    try:
        entries = [
            ("📁" if entry.is_dir() else "📄") + entry.name
            for entry in safe_path.iterdir()
            if not entry.name.startswith(".")
        ]
        return ok("\n".join(entries))
    except Exception as e:
        return err(f"Error listing directory: {e}")


def write_file_in_sandbox(tool_input: dict) -> dict:
    file_path: str = tool_input.get("file_path")
    content: str = tool_input.get("content", "")

    if content.strip() == "":
        return err("Content is empty, nothing to write, please provide some contents.")

    folder_check_result = pure_file_path_checker(file_path)

    if folder_check_result["is_error"]:
        return folder_check_result

    safe_path: Path = folder_check_result["content"]

    try:
        # Ensure the parent directory exists
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        isExistingFile = safe_path.is_file()

        with safe_path.open("a", encoding="utf-8") as f:
            f.write(("\n" if isExistingFile else "") + content)
        return ok(f"Successfully wrote to file: {file_path}")
    except Exception as e:
        print(f"Error writing to file: {file_path}: {e}")
        return err(f"Error writing to file: {e}")


def str_replace(tool_input: dict) -> dict:
    print(f"Received str_replace input: {tool_input}")
    path: str = tool_input.get("path")
    old_substr: str = tool_input.get("old_str", "")
    new_substr: str = tool_input.get("new_str", "")
    replace_all: bool = bool(tool_input.get("replace_all", False))

    # Validate inputs
    if old_substr == "":
        return err("Old substring is empty, please provide a valid old substring to replace.")
    if new_substr == "":
        return err("New substring is empty, please provide a valid new substring.")

    path_checker_result = file_path_checker(path)

    if path_checker_result["is_error"]:
        return path_checker_result

    safe_path: Path = path_checker_result["content"]

    try:
        content = safe_path.read_text(encoding="utf-8")
    except Exception as e:
        return err(f"Error reading file for replacement: {e}")

    occurrences = content.count(old_substr)
    relative_path = get_relative_path(safe_path, SANDBOX_PATH)

    if occurrences == 0:
        preview = content[:50] + "..." if len(content) > 50 else content
        return err(
            f"Cannot find the string to be replaced in {relative_path}. "
            f"String to be replaced preview: {preview}. "
            "Please check the string or block and try again."
        )

    if occurrences > 1 and not replace_all:
        previews = get_previews(content, old_substr)
        preview_text = "\n".join(previews[:5])  # Show up to 5 previews
        return err(
            f"Found {occurrences} occurrences of the string to be replaced in {relative_path}. "
            f"Here are some previews:\n{preview_text}\n"
            "Please modify your request to make the old_str unique and try again."
        )

    new_content = content.replace(old_substr, new_substr) if replace_all else content.replace(old_substr, new_substr, 1)

    try:
        safe_path.write_text(new_content, encoding="utf-8")
        return ok(
            f"Successfully replaced string in file: {get_relative_path(safe_path, SANDBOX_PATH)}, "
            f"from {old_substr} to {new_substr}. "
            f"File size: {len(new_content)} chars"
        )
    except Exception as e:
        return err(f"Error writing to file during string replacement: {e}")
