from .base import Message

class AIMessage(Message):
    role: str = "ai"
