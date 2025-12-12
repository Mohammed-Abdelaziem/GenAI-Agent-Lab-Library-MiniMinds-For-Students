from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field
from loguru import logger
import json
from langfuse import observe

from llm.base import LLMClient
from tools.registry import ToolRegistry


class BaseAgentState(BaseModel):
    """
    Holds the evolving state of an agent's execution.
    """
    messages: list[dict] = Field(default_factory=list)
    scratchpad: list[str] = Field(default_factory=list)
    is_finished: bool = False
    iteration: int = 0

    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        # extra for example tool_id
        msg.update(extra)
        self.messages.append(msg)


class ScratchpadAgentState(BaseAgentState):
    """
    Extends the base state with helper fields for the v2 agent
    (tracks written tests + pytest status).
    """
    test_files_written: set[str] = Field(default_factory=set)
    pytest_passed: bool = False
    recent_dir_signatures: list[str] = Field(default_factory=list)
    dir_listings_executed: int = 0
    target_module_read: bool = False
    read_files_seen: set[str] = Field(default_factory=set)


def prune_messages(
    messages: list[dict],
    keep_system: bool = True,
    keep_user: bool = True,
    last_n: int = 2,
    drop_tools: bool = False,
) -> list[dict]:
    """
    Keep minimal context: system + last user + last_n other messages.
    When drop_tools=True, tool role messages are removed before pruning
    so summaries can live in the scratchpad instead of raw tool outputs.
    """
    filtered_messages = (
        [m for m in messages if m.get("role") != "tool"] if drop_tools else messages
    )

    system_msgs = (
        [m for m in filtered_messages if m.get("role") == "system"]
        if keep_system
        else []
    )
    user_msgs = (
        [m for m in filtered_messages if m.get("role") == "user"] if keep_user else []
    )
    other = [m for m in filtered_messages if m.get("role") not in {"system", "user"}]
    trimmed_other = other[-last_n:] if last_n > 0 else []
    return system_msgs + user_msgs[-1:] + trimmed_other

class Agent(ABC):
    def __init__( self, llm: LLMClient, tool_registry: ToolRegistry, max_iterations: int = 100):
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        # initial state to start with
        self.inital_state = BaseAgentState()

    @abstractmethod
    def start_point(self, *args, **kwargs) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        raise NotImplementedError()
    
    @abstractmethod
    def run(self, state: BaseAgentState) -> BaseAgentState:
        """ Run 1 Step/Iteration """
        raise NotImplementedError()

    def iterate(self, *args, **kwargs) -> BaseAgentState:
        state = self.start_point(*args, **kwargs)
        while not state.is_finished and state.iteration < self.max_iterations:
            state.iteration += 1
            state = self.run(state)

        return state

    # LLM WRAPPER
    @observe(name="llm-call", as_type="generation")
    def llm_generate(self, state: BaseAgentState):
        tools = self.tool_registry.to_client_tools(self.llm.config.provider)
        return self.llm.generate(state.messages, tools=tools)
    # TOOL EXECUTION WRAPPER
    @observe(name="tool-call", as_type="tool")
    def call_tool(self, tool_call):
        """
        Execute a tool call safely with logging and error capture.
        tool_call shape:
        {
          "type": "function",
          "id": "...",
          "function": { "name": "...", "arguments": {... or json str ...} }
        }
        """
        if tool_call.get("type") != "function":
            return {"success": False, "error": f"Unsupported tool_call type {tool_call.get('type')}"}

        func_name = tool_call["function"]["name"]
        args_raw = tool_call["function"].get("arguments", {}) or {}

        if isinstance(args_raw, str):
            try:
                func_inputs = json.loads(args_raw)
            except Exception as e:
                return {"success": False, "error": f"Invalid JSON arguments: {e}"}
        else:
            func_inputs = args_raw

        try:
            func = self.tool_registry.get(func_name)
            if func is None:
                raise ValueError(f"Tool {func_name} not found")
            logger.debug(f"Calling tool {func_name} with {func_inputs}")
            result = func(**func_inputs)
            return {"success": True, "result": result}
        except Exception as e:
            logger.exception(f"Tool {func_name} failed")
            return {"success": False, "error": str(e)}
