"""A multi-turn LangGraph agent using interrupt()/resume for human-in-the-loop.

Same graph shape as compile.py, but the human node calls interrupt() to PAUSE
the graph and hand control back to the caller. The caller collects the next
message however it likes (here: input()) and resumes with Command(resume=...).
Each turn is therefore a separate invoke, which is the idiomatic LangGraph HIL
pattern and works from an API / web backend, not just a terminal.

Key differences vs compile.py:
  - needs a checkpointer (MemorySaver) + thread_id to remember where it paused;
  - the caller drives the loop, so there is no recursion_limit ceiling on turns.

Run from the project root as a module:

    uv run python -m lang_graph.compile_interrupt
"""

import uuid
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages.ai import add_usage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, GraphOutput, RunnableConfig, interrupt
from langsmith import ContextThreadPoolExecutor
from typing_extensions import TypedDict

from consts import SYSTEM_PROMPT_WITH_CACHE
from lang_graph import LG_TOOLS
from lang_graph.preview import LG_WRITE_PREVIEW_MAP
from lang_graph.tools import LG_TOOL_MAP

load_dotenv()

"""State Definition"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    pending_writes: list[ToolCall]  # store write_tool requests


"""Utils"""


def separate_read_and_write(tool_calls: list[ToolCall]) -> tuple[list[ToolCall], list[ToolCall]]:
    reads, writes = [], []
    for call in tool_calls:
        (writes if call["name"] in LG_WRITE_PREVIEW_MAP else reads).append(call)
    return reads, writes


def run_one_tool(tool_call: ToolCall) -> ToolMessage:
    id, name, args = destructToolCall(tool_call)
    tool_fn: BaseTool | None = LG_TOOL_MAP.get(name, None)

    if tool_fn is None:
        return ToolMessage(content="Tool is not found, please stop trying it again.", tool_call_id=id, status="error")

    try:
        return ToolMessage(content=str(tool_fn.invoke(args)), tool_call_id=id, status="success")
    # Just in case: directly raise interrupt exception
    except GraphBubbleUp:
        raise
    except Exception as e:
        # Use internal status to mark if tool is successful
        return ToolMessage(content=f"Error: {e}", tool_call_id=id, status="error")


def destructToolCall(tool_call: ToolCall) -> tuple[str, str, dict[str, Any]]:
    return tool_call["id"], tool_call["name"], tool_call["args"]


def print_total_usage(messages: list[AnyMessage]) -> None:
    """Sum usage_metadata across every AIMessage and print the totals.

    add_usage merges the nested input_token_details (cache_read / cache_creation)
    correctly, so the cache breakdown is preserved.
    """
    total = None
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            total = m.usage_metadata if total is None else add_usage(total, m.usage_metadata)

    if total is None:
        print("No usage data.")
        return

    details = total.get("input_token_details", {})
    print("=" * 40)
    print(f"Input tokens : {total['input_tokens']}")
    print(f"Output tokens: {total['output_tokens']}")
    print(f"Total tokens : {total['total_tokens']}")
    print(f"  cache_read    : {details.get('cache_read', 0)}")
    print(f"  cache_creation: {details.get('cache_creation', 0)}")
    print("=" * 40)


# 2. LLM node
llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024).bind_tools(LG_TOOLS)


"""Tool nodes"""


# read tools only
def read_tools_and_prepare_writes(state: AgentState):
    last = state["messages"][-1]

    if isinstance(last, AIMessage):
        reads, writes = separate_read_and_write(last.tool_calls)

        # run read calls in parallel
        with ContextThreadPoolExecutor(max_workers=5) as ex:
            return {"messages": [*ex.map(run_one_tool, reads)], "pending_writes": writes}
    else:
        # Unreachable
        # todo: How to elegantly raise error and stop the whole graph
        return {}


# Invoke interrupt and request permission from users
def preview_write_tools(state: AgentState) -> Command[Literal["preview_write_tools", "write_tools"]]:
    # Back to agent to handle tool results
    if len(state["pending_writes"]) == 0:
        return Command(goto="agent")

    write_tool = state["pending_writes"][0]
    id, name, args = destructToolCall(write_tool)

    preview_fn = LG_WRITE_PREVIEW_MAP.get(name, None)

    # Tool not found, loop itself and mention preview is not found
    if preview_fn is None:
        return Command(
            goto="preview_write_tools",
            update={
                "messages": [
                    ToolMessage(
                        content="Error: preview is not found, stop using this tool",
                        tool_call_id=id,
                        status="error",
                    )
                ],
                "pending_writes": state["pending_writes"][1:],
            },
        )

    try:
        preview = preview_fn(args)
        # Put the diff in the interrupt payload (not print): keeps it attached to the
        # approval prompt and avoids double-printing when the node re-runs on resume.
        approval: str = interrupt({"type": "preview", "print": f"\n{preview}\n\nApprove this edit? Y(es)/N(o): "})

        if approval.strip() in ["Y", "Yes"]:
            # Move to the execution node
            return Command(goto="write_tools")
        else:
            # Denied, loop itself and mention preview is denied by users
            return Command(
                goto="preview_write_tools",
                update={
                    "messages": [
                        ToolMessage(
                            content="Error: User denied this request, please do not try it again and ask what to do next.",
                            tool_call_id=id,
                            status="error",
                        )
                    ],
                    "pending_writes": state["pending_writes"][1:],
                },
            )

    except GraphBubbleUp:
        raise
    except Exception as e:
        # Preview runtime error, loop itself and mention preview fails during the runtime
        return Command(
            goto="preview_write_tools",
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=id,
                        status="error",
                    )
                ],
                "pending_writes": state["pending_writes"][1:],
            },
        )


# write tools only, required interrupt
def write_tools(state: AgentState) -> Command[Literal["preview_write_tools", "agent"]]:
    write_tool = state["pending_writes"][0]
    id, name, args = destructToolCall(write_tool)
    tool_fn: BaseTool | None = LG_TOOL_MAP.get(name, None)

    # It should not happen, just in case
    if tool_fn is None:
        return Command(
            goto="preview_write_tools",
            update={
                "messages": [
                    ToolMessage(
                        content="Error: tool is not found, stop using this tool",
                        tool_call_id=id,
                        status="error",
                    )
                ],
                "pending_writes": state["pending_writes"][1:],
            },
        )

    try:
        result = tool_fn.invoke(args)
        return Command(
            goto="preview_write_tools",
            update={
                "messages": [
                    ToolMessage(
                        content=str(result),
                        tool_call_id=id,
                        status="success",
                    )
                ],
                "pending_writes": state["pending_writes"][1:],
            },
        )
    except GraphBubbleUp:
        raise
    except Exception as e:
        return Command(
            goto="preview_write_tools",
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: {e}",
                        tool_call_id=id,
                        status="error",
                    )
                ],
                "pending_writes": state["pending_writes"][1:],
            },
        )


"""Agent node"""


def agent(state: AgentState):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_WITH_CACHE), *state["messages"]])
    # clear all pending write requests
    return {"messages": [response], "pending_writes": []}


"""Human node"""


# 4. Human node: interrupt() pauses the graph; whatever the caller passes as
#    Command(resume=...) becomes interrupt()'s return value and is appended.
def human(state: AgentState):
    text: str = interrupt({"type": "user", "print": "waiting for user input:"})

    # clear all pending write requests and start the next round of talk
    return {"messages": [("user", text)], "pending_writes": []}


"""Condition node"""


# 5. Conditional edge: tool calls -> tools; otherwise -> human for the next turn
def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "read_tools_and_prepare_writes" if isinstance(last, AIMessage) and last.tool_calls else "human"


# Router: Move to preview if pending_writes is not empty, otherwise back to Agent
def should_preview_or_return(state: AgentState):
    return "agent" if len(state["pending_writes"]) == 0 else "preview_write_tools"


# 6. Assemble the graph. interrupt() requires a checkpointer to persist state.
graph = (
    StateGraph(AgentState)
    .add_node("agent", agent)
    .add_node("read_tools_and_prepare_writes", read_tools_and_prepare_writes)
    .add_node("write_tools", write_tools)
    .add_node("preview_write_tools", preview_write_tools)
    .add_node("human", human)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue, ["human", "read_tools_and_prepare_writes"])
    .add_edge("human", "agent")
    .add_conditional_edges("read_tools_and_prepare_writes", should_preview_or_return, ["preview_write_tools", "agent"])
)

app = graph.compile(checkpointer=MemorySaver())


# 7. Run: the caller drives the loop, resuming the paused graph each turn.
if __name__ == "__main__":
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        # First turn seeds an initial user message; the graph runs until it pauses
        # at the human node's interrupt(). Input must be a state dict (wrap the message).
        result: GraphOutput[AgentState] = app.invoke(
            input={"messages": [HumanMessage("你好,你能做什么?")], "pending_writes": []}, config=config, version="v2"
        )

        # result.interrupts is empty when the graph finishes
        while result.interrupts:
            # Each interrupt carries its own prompt text in the payload ("print").
            # It will result in issue when parallel interrupts happening
            payload = result.interrupts[0].value

            # Only a user-input pause needs the agent's latest reply shown first;
            if payload.get("type") == "user":
                result.value["messages"][-1].pretty_print()

            text = input(f"{payload.get('print', 'You:')} ").strip()
            if not text or text.lower() == "exit":
                break

            # Resume from the interrupt; the text becomes the interrupt()'s return value.
            result = app.invoke(Command(resume=text), config=config, version="v2")
    except KeyboardInterrupt:
        print("\n^C Interrupted")
    finally:
        print("Bye.")
        print_total_usage(app.get_state(config).values.get("messages", []))
