"""A multi-turn LangGraph agent using input() for human-in-the-loop.

The graph loops: human -> agent <-> tools -> human -> ... until the user exits.
The human node reads the next message from the terminal with input(), so the
whole conversation runs inside a single app.invoke() call.

See compile_interrupt.py for the idiomatic interrupt()/resume version.

Run from the project root as a module:

    uv run python -m lang_graph.compile
"""

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict

from consts import SYSTEM_PROMPT_TXT
from lang_graph import LG_TOOLS

load_dotenv()


# 1. State: the running message history (add_messages appends instead of overwriting)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. LLM node: call the model with the tools bound, prepending the system prompt
llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=1024).bind_tools(LG_TOOLS)


def agent_node(state: AgentState):
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_TXT), *state["messages"]])
    return {"messages": [response]}


# 3. Human node: read the next user message from the terminal.
#    Returns a Command to route dynamically: 'exit' ends the graph, otherwise
#    the new message is appended and control goes back to the agent.
def human_node(state: AgentState) -> Command[Literal["agent", "__end__"]]:
    # Print the agent's latest reply before prompting for the next turn.
    # On the very first turn the history is empty, so there's nothing to print.
    if state["messages"]:
        last = state["messages"][-1]
        if getattr(last, "content", None):
            print(f"\nAgent: {last.content}\n")

    text = input("You: ").strip()
    if not text or text.lower() == "exit":
        return Command(goto=END)
    return Command(goto="agent", update={"messages": [("user", text)]})  # HumanMessage(content=text)


# 4. Conditional edge: if the model asked for a tool, run it; otherwise hand off
#    to the human for the next turn (instead of ending).
def should_continue(state: AgentState):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "human"


# 5. Assemble the graph: human -> agent <-> tools -> human loop
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(LG_TOOLS))
graph.add_node("human", human_node)
graph.set_entry_point("human")  # ask the user first, no need to seed a message
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
# "human" routes via Command(goto=...), so it needs no static edge.
app = graph.compile()


# 6. Run: one invoke drives the whole multi-turn session.
#    recursion_limit must be raised because every turn consumes several supersteps
#    and the default (25) would cut the conversation off after a few turns.
if __name__ == "__main__":
    app.invoke({"messages": []}, config={"recursion_limit": 1000})
    print("Bye.")
