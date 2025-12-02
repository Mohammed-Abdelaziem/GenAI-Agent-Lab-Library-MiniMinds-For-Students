from .base import Message

class HumanMessage(Message):
    role: str = "human"
