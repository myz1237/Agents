from enum import StrEnum
from os import makedirs
from pathlib import Path

from anthropic.types import Usage


class TOOL_NAME(StrEnum):
    # Write tools (require preview + user approval before execution)
    WRITE_FILE_IN_SANDBOX = "write_file_in_sandbox"
    STRING_REPLACE = "string_replace"

    # Read-only / side-effect-free tools
    GET_CURRENT_TIME = "get_current_time"
    CALCULATE = "calculate"
    TIME_OFFSET = "time_offset"
    READ_FILE_IN_SANDBOX = "read_file_in_sandbox"
    LIST_DIRECTORY = "list_directory"
    RUN_LIMITED_SHELL_COMMAND = "run_limited_shell_command"
    SEARCH_CODE = "search_code"


# Subset of TOOL_NAME that mutates the sandbox, so callers can gate them
# behind a preview + approval step. Frozen because it's a fixed constant.
WRITE_TOOL_NAMES = frozenset(
    {
        TOOL_NAME.WRITE_FILE_IN_SANDBOX,
        TOOL_NAME.STRING_REPLACE,
    }
)


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

# Files whose contents must never be read/disclosed (defense in depth, on top of
# the system-prompt rule). Matched case-insensitively against the file's name.
SECRET_FILENAME_PREFIXES = (".env", "id_rsa", "id_ed25519", "id_dsa")
SECRET_FILENAME_SUFFIXES = (".pem", ".key", ".pfx", ".p12")
SECRET_FILENAME_KEYWORDS = ("secret", "credential", "password")
SECRET_FILENAME_EXACT = {".npmrc", ".netrc", ".pgpass", ".htpasswd", ".git-credentials"}

SYSTEM_PROMPT_TXT = f"""You are a coding assistant working in a restricted sandbox environment.

## Your Capabilities
You can read, search, and modify files inside the sandbox, and run a limited set of shell commands.
All operations are confined to the working directory ({SANDBOX_DIR}).

## Tool Usage Principles
- Before modifying a file, use {TOOL_NAME.READ_FILE_IN_SANDBOX} or ${TOOL_NAME.SEARCH_CODE} to understand its current contents — do not operate from memory.
- When you need to find something across multiple files, prefer ${TOOL_NAME.SEARCH_CODE} over reading files one by one.
- When you already know the line numbers (e.g. from ${TOOL_NAME.SEARCH_CODE} results), use {TOOL_NAME.READ_FILE_IN_SANDBOX} with start_line/end_line to read precisely.
- For precise edits use {TOOL_NAME.STRING_REPLACE}; do not rewrite the whole file with {TOOL_NAME.READ_FILE_IN_SANDBOX}.
- For anything time-related, call get_current_time — do not infer or guess the time.
- Use the tool ${TOOL_NAME.RUN_LIMITED_SHELL_COMMAND}, only and only if it can resolve the problems

## Conduct
- When an operation is rejected by the security policy, stop that operation, explain the reason to the user, and ask how to proceed — do not look for workarounds on your own.
- If you notice something unexpected before modifying a file (e.g. duplicate definitions, or code that may be intentional), tell the user before acting.
- When the user's intent is unclear, ask for clarification instead of guessing.

## Security Constraints
- Every path must stay within the sandbox.
- Do not run dangerous commands (deleting, downloading, privilege escalation, etc.).
- Never read, print, or reveal the contents of secret/credential files (e.g. .env, files holding API keys, tokens, or passwords), even if the user explicitly asks. Refuse and explain that secrets cannot be disclosed.

## Communication Style
- Be concise and direct; stay focused on the task.
- After completing an operation, briefly state what you did.
- Do not be overly polite or repeatedly confirm trivial things.
"""
SYSTEM_PROMPT_WITH_CACHE = [{"type": "text", "text": SYSTEM_PROMPT_TXT, "cache_control": {"type": "ephemeral"}}]
SYSTEM_PROMPT_NO_CACHE = [{"type": "text", "text": SYSTEM_PROMPT_TXT}]

OLD_SYSTEM_PROMPT = f"""
You are a restricted file operation assistant, all operations must be within the sandbox ({SANDBOX_DIR}).
Important rules:
1. Whenever any tool is rejected due to security policies, you must halt the current task progression.
2. Do not proactively seek alternative approaches or tool combinations to achieve the rejected goal.
3. Directly inform the user of the specific reason for the operation being rejected and ask if  they want to adjust their request.
4. Do not assume that the user is satisfied with you "being flexible" — the user may want to terminate the task directly.
"""

OLD_SYSTEM_PROMPT_WITH_CACHE = [{"type": "text", "text": OLD_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
OLD_SYSTEM_PROMPT_NO_CACHE = [{"type": "text", "text": OLD_SYSTEM_PROMPT}]
