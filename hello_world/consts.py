from enum import StrEnum
from os import makedirs
from pathlib import Path

from anthropic.types import Usage


class TOOL_NAME(StrEnum):
    GET_CURRENT_TIME = "get_current_time"
    CALCULATE = "calculate"
    TIME_OFFSET = "time_offset"
    READ_FILE_IN_SANDBOX = "read_file_in_sandbox"
    LIST_DIRECTORY = "list_directory"
    WRITE_FILE_IN_SANDBOX = "write_file_in_sandbox"
    RUN_LIMITED_SHELL_COMMAND = "run_limited_shell_command"
    STRING_REPLACE = "string_replace"


DEFAULT_MAX_ITERATION = 100
# Anchor the sandbox to this file's location, not the current working
# directory, so it resolves to the same place no matter where the script is run.
SANDBOX_PATH = Path(__file__).resolve().parent / "sandbox"
SANDBOX_DIR = str(SANDBOX_PATH)
makedirs(SANDBOX_DIR, exist_ok=True)

EMPTY_USAGE = Usage(
    cache_creation=None,
    cache_creation_input_tokens=None,
    cache_read_input_tokens=None,
    inference_geo=None,
    input_tokens=0,
    output_tokens=0,
    output_tokens_details=None,
    server_tool_use=None,
    service_tier=None,
)

ALLOWED_COMMANDS = {"ls", "cat", "grep", "find", "wc", "head", "tail", "echo", "pwd"}

SYSTEM_PROMPT_TXT = f"""
You are a restricted file operation assistant, all operations must be within the sandbox ({SANDBOX_DIR}).
Important rules:
1. Whenever any tool is rejected due to security policies, you must halt the current task progression.
2. Do not proactively seek alternative approaches or tool combinations to achieve the rejected goal.
3. Directly inform the user of the specific reason for the operation being rejected and ask if  they want to adjust their request.
4. Do not assume that the user is satisfied with you "being flexible" — the user may want to terminate the task directly.
"""
SYSTEM_PROMPT_WITH_CACHE = [{"type": "text", "text": SYSTEM_PROMPT_TXT, "cache_control": {"type": "ephemeral"}}]
SYSTEM_PROMPT_NO_CACHE = [{"type": "text", "text": SYSTEM_PROMPT_TXT}]
