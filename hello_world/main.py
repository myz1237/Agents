import os
import uuid

import anthropic
from anthropic.types import Model, Usage
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

from consts import DEFAULT_MAX_ITERATION, EMPTY_USAGE, SYSTEM_PROMPT_WITH_CACHE, TOOL_NAME
from tools import tool_map, tools
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
    # Add session id for Langfuse to group one chat in the same session together.
    with propagate_attributes(session_id=session_id):
        for i in range(max_iteration):
            print(f"Iteration {i + 1}")

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT_WITH_CACHE,
                tools=tools,
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
