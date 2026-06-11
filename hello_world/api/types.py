from pydantic import BaseModel


# Data Validation
class AgentRequest(BaseModel):
    message: str
    session_id: str | None = None


class AgentResponse(BaseModel):
    session_id: str
    response: str
    iterations: int
