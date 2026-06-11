import uuid

from fastapi import FastAPI

from agent import run_agent
from api.types import AgentRequest, AgentResponse
from eval.result import AgentResult

api = FastAPI(title="Coding Agent API", version="0.1.0")
SESSIONS: dict[str, list] = {}


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/agent/run", response_model=AgentResponse)
def run_agent_endpoint(req: AgentRequest):
    session_id = req.session_id or str(uuid.uuid4())
    # setdefault stores the (possibly new) history list in SESSIONS; run_agent
    # appends turns to it in place, so the next request with the same
    # session_id continues the conversation.
    history = SESSIONS.setdefault(session_id, [])
    # eval_mode=True: no interactive user behind an HTTP request, so the agent
    # must stop at end_turn instead of blocking on input() / write previews.
    _, messages, iterations = run_agent(req.message, session_id, max_iteration=20, eval_mode=True, history=history)
    return AgentResponse(
        session_id=session_id,
        response=AgentResult(messages).final_text(),
        iterations=iterations,
    )
