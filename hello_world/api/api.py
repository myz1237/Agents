import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent import agent_stream_generator, run_agent
from api.types import AgentRequest, AgentResponse
from eval.result import AgentResult
from sessions import SESSIONS

api = FastAPI(title="Coding Agent API", version="0.1.0")


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


@api.post("/agent/stream")
def run_agent_stream_endpoint(req: AgentRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(agent_stream_generator(req.message, session_id=session_id), media_type="text/event-stream")
