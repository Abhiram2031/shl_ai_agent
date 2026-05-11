from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

class HealthResponse(BaseModel):
    status: str