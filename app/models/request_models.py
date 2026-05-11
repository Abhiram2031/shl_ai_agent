from pydantic import BaseModel, field_validator
from typing import List

VALID_ROLES = {"user", "assistant"}

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in VALID_ROLES:
            raise ValueError(
                "Invalid role detected. Only 'user' and 'assistant' roles are allowed."
            )
        return v


class ChatRequest(BaseModel):
    messages: List[Message]