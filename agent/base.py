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
    is_finished: bool = False
    iteration: int = 0

    def add_message(self, role: str, content: str, **extra):
        msg = {"role": role, "content": content}
        # extra for example tool_id
        msg.update(extra)
        self.messages.append(msg)
        

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
