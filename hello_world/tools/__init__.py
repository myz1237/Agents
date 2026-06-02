"""Tool package.

Only `tool_map` is the public interface — import it elsewhere as:

    from tools import tool_map

The individual tool functions live in sibling modules (times, files,
calcualtion_eval) and are wired together here. Keeping the wiring in
__init__.py means callers never need to know which file a tool lives in.
"""

from consts import TOOL_NAME

from .calcualtion_eval import calculator
from .descriptions import tools
from .files import code_search, list_directory, read_file_in_sandbox, str_replace, write_file_in_sandbox
from .shell import run_shell_in_sandbox
from .times import get_current_time, time_offset

# name (from the TOOL_NAME enum) -> the function that implements it
tool_map = {
    TOOL_NAME.GET_CURRENT_TIME: get_current_time,
    TOOL_NAME.CALCULATE: calculator,
    TOOL_NAME.TIME_OFFSET: time_offset,
    TOOL_NAME.READ_FILE_IN_SANDBOX: read_file_in_sandbox,
    TOOL_NAME.LIST_DIRECTORY: list_directory,
    TOOL_NAME.WRITE_FILE_IN_SANDBOX: write_file_in_sandbox,
    TOOL_NAME.RUN_LIMITED_SHELL_COMMAND: run_shell_in_sandbox,
    TOOL_NAME.STRING_REPLACE: str_replace,
    TOOL_NAME.SEARCH_CODE: code_search,
}

# Controls what `from tools import *` exposes — only the map, nothing else.
__all__ = ["tool_map", "tools"]
