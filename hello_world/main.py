import os
import uuid

import anthropic
from anthropic.types import Model, Usage
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

from consts import DEFAULT_MAX_ITERATION, EMPTY_USAGE, SYSTEM_PROMPT_WITH_CACHE, TOOL_NAME
from tools import tool_map
from utils import err, print_usage, usage_add

load_dotenv()
AnthropicInstrumentor().instrument()
langfuse_client = get_client()

if langfuse_client.auth_check():
    print("Successfully authenticated with Langfuse.")
else:
    print("Failed to authenticate with Langfuse. Please check your API key and base URL.")


api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

client = anthropic.Anthropic(api_key=api_key)

# Model configuration
model: Model = "claude-sonnet-4-5"
max_tokens: int = 1024


def hello_world() -> None:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT_WITH_CACHE,
        messages=[{"role": "user", "content": "Tell me in one sentence why the sky is blue?"}],
    )
    print(response.content)


def get_tools() -> list[dict]:
    tools = [
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
    return tools


def is_tool_available(tool_name: TOOL_NAME) -> bool:
    return tool_name in tool_map


@observe(name="execute_tool")
def execute_tool(tool_name: TOOL_NAME, tool_input: dict) -> dict:
    print(f"Executing tool: {tool_name} with input: {tool_input}")
    func = tool_map.get(tool_name)
    if not func:
        return err(f"Unknown tool: {tool_name!r}")
    return func(tool_input)


@observe(name="run_agent")
def run_agent(
    user_message: str,
    session_id: str,
    max_iteration: int = DEFAULT_MAX_ITERATION,
) -> Usage:
    cumulative_usages = EMPTY_USAGE
    messages = [{"role": "user", "content": user_message}]
    with propagate_attributes(session_id=session_id):
        for i in range(max_iteration):
            print(f"Iteration {i + 1}")

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT_WITH_CACHE,
                tools=get_tools(),
                messages=messages,
            )

            cumulative_usages = usage_add(cumulative_usages, response.usage)

            stop_reason = response.stop_reason
            print(f"[stop reason]: {response.stop_reason}")
            messages.append({"role": "assistant", "content": response.content})

            if stop_reason == "max_tokens":
                print("Model has run out of max tokens. Stopping.")
                return cumulative_usages

            if stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        print(f"Final answer: {block.text}")
                user_input = input("Enter user message (or 'exit' to quit): ")
                if not user_input.strip() or user_input.lower() == "exit":
                    print("Exiting.")
                    return cumulative_usages
                messages.append({"role": "user", "content": user_input})
                continue

            if stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if not is_tool_available(block.name):
                            print(f"Tool {block.name} is not available. Skipping.")
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    # Do not return any tool information here, otherwise users would see it
                                    "content": "Tool is not available.",
                                    "is_error": True,
                                }
                            )
                        else:
                            print(f"Invoking tool: {block.name}")
                            tool_result = execute_tool(block.name, block.input)
                            print(f"Tool result: {tool_result}")
                            # Append the tool result to the messages so that the model can see
                            # the result in the next iteration
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": tool_result.get("content"),
                                    "is_error": tool_result.get("is_error"),
                                }
                            )

                messages.append({"role": "user", "content": tool_results})
                continue

            print(f"Unknown stop reason. Stopping. {stop_reason}")
            continue
        print("Max iteration reached. Stopping.")
    return cumulative_usages


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    cumulative_usages = run_agent("列一下sandbox下面的文件, 并读取mini-project的内容", session_id)
    print_usage(cumulative_usages)
    langfuse_client.flush()
