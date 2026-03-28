from pydantic import BaseModel, Field
from typing import Any

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 8
    debug: bool = False

class AskResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


