"""LangGraph 版本的沙箱编码 Agent —— 图的定义。

这是 agent.py(手写 for 循环 + Anthropic 原生 API)的 LangGraph 重写版。
工具实现搬到了 lang_graph/ 包里(原生 @tool + 自定义异常),本文件只负责
"图"本身:State、节点、条件边、组装。

对应关系:
    agent.py 里的 for 循环              -> StateGraph 的 agent <-> tools 环
    agent.py 里 stop_reason == tool_use -> should_continue 条件边
    agent.py 里 execute_tool(TOOL_MAP)  -> ToolNode(LC_TOOLS)
    agent.py 里调用方持有的 history     -> MemorySaver checkpointer(按 thread_id)

运行(在项目根目录):
    uv run python langgrap_version.py
"""

from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from consts import SYSTEM_PROMPT_TXT
from lang_graph import LG_TOOLS

load_dotenv()

# Model configuration —— 与 agent.py 保持一致
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# State —— 一条会自动累加消息的对话历史(对应 agent.py 的 messages)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# LLM 节点(对应 agent.py 里 client.messages.create 那段)
# 系统提示词每轮临时拼到最前面传给模型,但不写进 state,避免重复堆积。
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS).bind_tools(LG_TOOLS)


def agent_node(state: AgentState) -> dict:
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_TXT), *state["messages"]])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 条件边(对应 agent.py 里 stop_reason == "tool_use" 的判断)
# 模型最后一条消息带 tool_calls -> 去执行工具;否则结束。
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ---------------------------------------------------------------------------
# 组装图(对应 agent.py 里的 for 循环)
# MemorySaver 让同一个 thread_id 的多次 invoke 自动续上对话历史,
# 等价于 agent.py 里调用方持有并复用的 history / SESSIONS。
# ToolNode 默认会捕获工具抛出的自定义异常,转成 status=error 的 ToolMessage 回传模型。
# ---------------------------------------------------------------------------
def build_app():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(LG_TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")  # 工具执行完回到 agent
    return graph.compile(checkpointer=MemorySaver())


app = build_app()


def main() -> None:
    """交互式 REPL,对应 agent.py 里 run_agent 的人机循环。"""
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("LangGraph agent ready. 输入 exit 退出。\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "exit":
            print("Bye.")
            break
        result = app.invoke({"messages": [("user", user_input)]}, config=config)
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
