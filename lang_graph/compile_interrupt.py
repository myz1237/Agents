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
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from consts import SYSTEM_PROMPT_TXT
from lang_graph import LG_TOOLS

load_dotenv()


# 1. State: the running message history
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_writes: list  # store write_tool requests


# 2. LLM node
llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024).bind_tools(LG_TOOLS)


"""Tool nodes"""


# read tools only
def read_tool_node(state: AgentState):
    print(state)


# Invoke interrupt and request permission from users
def preview_write_tool_node(state: AgentState):
    print(state)


# write tools only, required interrupt
def write_tool_node(state: AgentState):
    print(state)


"""Agent node"""


def agent_node(state: AgentState):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_TXT), *state["messages"]])
    return {"messages": [response]}


"""Human node"""


# 4. Human node: interrupt() pauses the graph; whatever the caller passes as
#    Command(resume=...) becomes interrupt()'s return value and is appended.
def human_node(state: AgentState):
    text = interrupt("waiting for user input")
    return {"messages": [("user", text)]}


"""Condition node"""


# 5. Conditional edge: tool calls -> tools; otherwise -> human for the next turn
def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "human"


def should_write_after_read(state: AgentState):
    print(state)


def should_write_after_write(state: AgentState):
    print(state)


# 6. Assemble the graph. interrupt() requires a checkpointer to persist state.
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools=LG_TOOLS))
graph.add_node("human", human_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
graph.add_edge("human", "agent")  # after the user replies, go back to the agent
app = graph.compile(checkpointer=MemorySaver())


# 7. Run: the caller drives the loop, resuming the paused graph each turn.
if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First turn seeds an initial user message; the graph runs until it pauses
    # at the human node's interrupt().
    app.invoke({"messages": [("user", "你好,你能做什么?")]}, config)

    while True:
        # At the pause point the last message is the agent's reply.
        print(f"\nAgent: {app.get_state(config).values['messages'][-1].content}\n")

        text = input("You: ").strip()
        if not text or text.lower() == "exit":
            print("Bye.")
            break

        # Resume from the interrupt; the text becomes the human node's output.
        app.invoke(Command(resume=text), config)
