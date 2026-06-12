import json
import os

import anthropic
from anthropic.types import Model
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

from consts import DEFAULT_MAX_ITERATION, EMPTY_USAGE, SYSTEM_PROMPT_WITH_CACHE, TOOL_NAME
from sessions import SESSIONS
from tools import TOOL_MAP, WRITE_PREVIEW_MAP, tools
from utils import err, usage_add

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
    return tool_name in TOOL_MAP


def is_write_preview_tool(tool_name: TOOL_NAME) -> bool:
    return tool_name in WRITE_PREVIEW_MAP


def confirm_write(preview_result: str) -> str:
    print(f"\n{'=' * 60}")
    print("⚠️  Agent requests file writes:")
    print(f"{'─' * 60}")
    print(preview_result)
    print(f"{'=' * 60}")

    answer = input("Confirm to execute? [y]es / [n]o / [a]lways(No further ask in this session): ").strip().lower()
    return answer


# Only for LLM, not real LLM Tool Execution
# Give more insights to users when modification happens
def execute_preview_tool(tool_name: TOOL_NAME, tool_input: dict) -> dict:
    print(f"Executing preview tool: {tool_name} with input: {tool_input}")
    func = WRITE_PREVIEW_MAP.get(tool_name)
    if not func:
        return err(f"Unknown preview tool: {tool_name!r}")
    return func(tool_input)


@observe(name="execute_tool")
def execute_tool(tool_name: TOOL_NAME, tool_input: dict) -> dict:
    print(f"Executing tool: {tool_name} with input: {tool_input}")
    func = TOOL_MAP.get(tool_name)
    if not func:
        return err(f"Unknown tool: {tool_name!r}")
    return func(tool_input)


@observe(name="run_agent")
def run_agent(
    user_message: str,
    session_id: str,
    max_iteration: int = DEFAULT_MAX_ITERATION,
    eval_mode: bool = False,
    history: list | None = None,
) -> tuple:
    cumulative_usages = EMPTY_USAGE
    # The caller owns the history list (e.g. the API's session store); new
    # turns are appended to it in place, so the next call sees the full chat.
    messages = history if history is not None else []
    messages.append({"role": "user", "content": user_message})
    auto_approve = False  # Only used for write tools
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
                return cumulative_usages, messages, i + 1

            if stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        print(f"Final answer: {block.text}")
                # In eval mode there is no interactive user: stop as soon as the
                # agent finishes a turn instead of blocking on input().
                if eval_mode:
                    return cumulative_usages, messages, i + 1
                user_input = input("Enter user message (or 'exit' to quit): ")
                if not user_input.strip() or user_input.lower() == "exit":
                    print("Exiting.")
                    return cumulative_usages, messages, i + 1
                messages.append({"role": "user", "content": user_input})
                continue

            if stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        if not is_tool_available(tool_name):
                            print(f"Tool {tool_name} is not available. Skipping.")
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    # Do not return any tool information here, otherwise users would see it
                                    "content": "Tool is not available.",
                                    "is_error": True,
                                }
                            )
                            continue
                        else:
                            # Skip preview when eval mode
                            if not eval_mode and (is_write_preview_tool(tool_name) and not auto_approve):
                                print(f"Invoking preview tool: {tool_name}")
                                preview_tool_result = execute_preview_tool(tool_name, block.input)
                                # preview tool error
                                if preview_tool_result["is_error"]:
                                    tool_results.append(
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": f"{tool_name} preview failed, please try it again.",
                                            "is_error": True,
                                        }
                                    )
                                    continue
                                else:
                                    # Gather user's response
                                    answer = confirm_write(preview_tool_result["content"])
                                    if answer in ("a", "always"):
                                        auto_approve = True
                                    elif answer not in ("y", "yes", ""):
                                        tool_results.append(
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": block.id,
                                                "content": (
                                                    "User has denied this request, "
                                                    "please do not try again and ask user how to adjust."
                                                ),
                                                "is_error": True,
                                            }
                                        )
                                        continue

                            print(f"Invoking tool: {tool_name}")
                            tool_result = execute_tool(tool_name, block.input)
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
    return cumulative_usages, messages, i + 1


def learn_run_agent_with_stream(
    user_message: str,
    session_id: str,
    max_iteration: int = DEFAULT_MAX_ITERATION,
):
    message = ""
    json_str = ""
    messages = [{"role": "user", "content": user_message}]
    current_type = ""
    iteration = 0
    with client.messages.stream(
        model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT_WITH_CACHE, tools=tools, messages=messages
    ) as stream:
        iteration += 1
        for event in stream:
            if iteration > max_iteration:
                break
            if event.type == "message_start":
                print(f"Message starts: {(event.message)}")
            if event.type == "message_stop":
                print(f"All message end: {(event.message)}")
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    print(f"Start to call tool: {event.content_block.name}")
                else:
                    print(f"{event.index + 1} Message is coming...")
            elif event.type == "thinking":
                print("Start thinking")
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    current_type = "text"
                    print(f"Text output: {event.delta.text}")
                    message += event.delta.text
                elif event.delta.type == "input_json_delta":
                    current_type = "json"
                    print(f"JSON output: {event.delta.partial_json}")
                    json_str += event.delta.partial_json
                elif event.delta.type == "thinking_delta":
                    print(f"Thinking output: {event.delta.thinking}")
            elif event.type == "content_block_stop":
                print(f"The whole message: {message if current_type == 'text' else (json_str)}")
                message = ""  # Clear messages
                json_str = ""  # Clear json


def agent_stream_generator(user_message: str, session_id: str, max_iteration: int = DEFAULT_MAX_ITERATION):
    messages = SESSIONS.get(session_id, [])
    messages.append({"role": "user", "content": user_message})

    for i in range(max_iteration):
        print(f"Iteration {i + 1}")
        with client.messages.stream(
            model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT_WITH_CACHE, tools=tools, messages=messages
        ) as stream:
            # Only for stream requests. Text arrives piece by piece on
            # content_block_delta events; content_block_start only tells us a
            # block (e.g. a tool call) is beginning.
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "tool_use":
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': event.content_block.name})}\n\n"
                elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"
            # Same to normal chat mode
            final_message = stream.get_final_message()

        messages.append({"role": "assistant", "content": final_message.content})
        stop_reason = final_message.stop_reason

        if stop_reason == "max_tokens":
            print("Model has run out of max tokens. Stopping.")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Max token reached'})}\n\n"
            break

        if stop_reason == "end_turn":
            # The text was already streamed delta-by-delta above; emitting
            # final_message.content again would send the answer twice.
            break  # Jump out of the loop and return end data
        if stop_reason == "tool_use":
            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    if not is_tool_available(tool_name):
                        print(f"Tool {tool_name} is not available. Skipping.")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                # Do not return any tool information here, otherwise users would see it
                                "content": "Tool is not available.",
                                "is_error": True,
                            }
                        )
                        continue
                    else:
                        print(f"Invoking tool: {tool_name}")
                        tool_result = execute_tool(tool_name, block.input)
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
                        continue

            messages.append({"role": "user", "content": tool_results})
            continue
    SESSIONS[session_id] = messages
    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
