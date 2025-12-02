from .base import Message

class ToolMessage(Message):
    role: str = "tool"
    tool_name: str = ""
