"""This module defines the descriptions and input schemas of tools that the agent can use.
The descriptions and input schemas are used by the agent to decide when and how to use the tools.
"""

from consts import TOOL_NAME

tools: list[dict] = [
    # {
    #     "name": TOOL_NAME.GET_CURRENT_TIME,
    #     "description": ("Get the current time. Invoke it when user asks for the current time."),
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "timezone": {
    #                 "type": "string",
    #                 "description": (
    #                     "The timezone to get the current time for, as an "
    #                     "IANA name like 'Asia/Shanghai' or 'UTC'. "
    #                     "Default is Asia/Shanghai (UTC+8)."
    #                 ),
    #             }
    #         },
    #         "required": [],
    #     },
    # },
    # {
    #     "name": TOOL_NAME.CALCULATE,
    #     "description": (
    #         "Execute pure numerical operations (addition, subtraction, multiplication, division, exponentiation, "
    #         "square root, etc.). Do not use for time calculations; use time_offset for that."
    #     ),
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "expression": {
    #                 "type": "string",
    #                 "description": ("The mathematical expression to calculate, like 2 * 5 + 1"),
    #             }
    #         },
    #         "required": ["expression"],
    #     },
    # },
    # {
    #     "name": TOOL_NAME.TIME_OFFSET,
    #     "description": "Calculate a time offset. Invoke it when user asks to calculate a time offset.",
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "base_time": {
    #                 "type": "string",
    #                 "description": "The base time, in ISO format, e.g., '2026-05-29 23:30:24'",
    #             },
    #             "offset_seconds": {
    #                 "type": "integer",
    #                 "description": "The offset in seconds, positive for later, negative for earlier",
    #             },
    #         },
    #         "required": ["base_time", "offset_seconds"],
    #     },
    # },
    {
        "name": TOOL_NAME.READ_FILE_IN_SANDBOX,
        "description": "Read the contents of a file. Invoke it when user asks to read a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute or relative path to the file to read.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "The maximum number of lines to read from the file. Default is 500.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": TOOL_NAME.LIST_DIRECTORY,
        "description": "List the contents of a directory. Invoke it when user asks to list a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "The absolute or relative path to the directory to list.",
                }
            },
            "required": ["dir_path"],
        },
    },
    {
        "name": TOOL_NAME.WRITE_FILE_IN_SANDBOX,
        "description": "Write content to a file. Invoke it when user asks to write to a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute or relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    # {
    #     "name": TOOL_NAME.RUN_LIMITED_SHELL_COMMAND,
    #     "description": (
    #         "Run a limited shell command. Invoke it when user asks to run a shell command. Only a limited"
    #         f" set of safe commands are allowed ({', '.join(ALLOWED_COMMANDS)})."
    #     ),
    #     "input_schema": {
    #         "type": "object",
    #         "properties": {
    #             "command": {
    #                 "type": "string",
    #                 "description": "The shell command to run.",
    #             }
    #         },
    #         "required": ["command"],
    #     },
    # },
    {
        "name": TOOL_NAME.STRING_REPLACE,
        "description": (
            "Replace strings in a file. Invoke it when user asks to replace strings in a file. "
            "If you find multiple occurrences of the string to be replaced, "
            "ask users which one to replace, otherwise no action will be taken. "
            "Please make the old_str unique, multiple occurrences of the same old_str will be denied for safety. "
            "It's recommended to modify parts of contents of a file, more efficient than the tool "
            f"combination of {TOOL_NAME.READ_FILE_IN_SANDBOX} + {TOOL_NAME.WRITE_FILE_IN_SANDBOX}, "
            "If you wanna append new content, new_str should be old_str plus the new content. "
            "Do not add all contents as old_str, only pick up what you wanna change"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to replace strings in.",
                },
                "old_str": {
                    "type": "string",
                    "description": "The string to be replaced.",
                },
                "new_str": {
                    "type": "string",
                    "description": "The string to replace with.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "Set true to replace all occurrences at once. "
                        "When multiple occurrences exist and the user wants all of them changed, "
                        "PREFER setting replace_all=true with a minimal old_str, rather than "
                        "expanding old_str to cover a large block. Default false."
                    ),
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
]
