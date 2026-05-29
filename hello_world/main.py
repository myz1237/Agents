import os

import anthropic
from anthropic.types import Model
from dotenv import load_dotenv

from consts import TOOL_NAME
from tools import tool_map

load_dotenv()

api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

client = anthropic.Anthropic(api_key=api_key)

# Model configuration
model: Model = "claude-haiku-4-5"
max_tokens: int = 1024
tokens_used: dict = {
    "input_tokens": 0,
    "output_tokens": 0,
}


def hello_world() -> None:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": "Tell me in one sentence why the sky is blue?"}],
    )
    print(response.content)


def get_tools() -> list[dict]:
    tools = [
        {
            "name": TOOL_NAME.GET_CURRENT_TIME,
            "description": ("Get the current time. Invoke it when user asks for the current time."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "The timezone to get the current time for, as an "
                            "IANA name like 'Asia/Shanghai' or 'UTC'. "
                            "Default is Asia/Shanghai (UTC+8)."
                        ),
                    }
                },
                "required": [],
            },
        },
        {
            "name": TOOL_NAME.CALCULATE,
            "description": (
                "Execute pure numerical operations (addition, subtraction, multiplication, division, exponentiation, "
                "square root, etc.). Do not use for time calculations; use time_offset for that."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": ("The mathematical expression to calculate, like 2 * 5 + 1"),
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": TOOL_NAME.TIME_OFFSET,
            "description": "Calculate a time offset. Invoke it when user asks to calculate a time offset.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "base_time": {
                        "type": "string",
                        "description": "The base time, in ISO format, e.g., '2026-05-29 23:30:24'",
                    },
                    "offset_seconds": {
                        "type": "integer",
                        "description": "The offset in seconds, positive for later, negative for earlier",
                    },
                },
                "required": ["base_time", "offset_seconds"],
            },
        },
    ]
    return tools


def execute_tool(tool_name: TOOL_NAME, tool_input: dict) -> dict:
    func = tool_map.get(tool_name)
    if not func:
        return {"content": f"Unknown tool: {tool_name!r}", "is_error": True}
    return func(tool_input)


def run_agent(user_message: str, max_iteration: int = 10) -> None:
    messages = [{"role": "user", "content": user_message}]
    for i in range(max_iteration):
        print(f"Iteration {i + 1}")

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=get_tools(),
            messages=messages,
        )

        tokens_used["input_tokens"] += response.usage.input_tokens
        tokens_used["output_tokens"] += response.usage.output_tokens
        stop_reason = response.stop_reason
        print(f"[stop reason]: {response.stop_reason}")
        messages.append({"role": "assistant", "content": response.content})

        if stop_reason == "max_tokens":
            print("Model has run out of max tokens. Stopping.")
            return

        if stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"Final answer: {block.text}")
            user_input = input("Enter user message (or 'exit' to quit): ")
            if not user_input.strip() or user_input.lower() == "exit":
                print("Exiting.")
                return
            messages.append({"role": "user", "content": user_input})
            continue

        if stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if block.name == TOOL_NAME.GET_CURRENT_TIME:
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
                    if block.name == TOOL_NAME.CALCULATE:
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
                    if block.name == TOOL_NAME.TIME_OFFSET:
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


if __name__ == "__main__":
    run_agent("现在东京时间是几点？过去 17 小时 43 分钟 28 秒前是几点？")
    print(f"Input tokens used: {tokens_used['input_tokens']}")
    print(f"Output tokens used: {tokens_used['output_tokens']}")
