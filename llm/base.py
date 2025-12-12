from abc import ABC, abstractmethod
from typing import Iterator, Optional, Any, List

from .config import LLMConfig

try:
    from langfuse import observe, get_client  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    observe = None  # type: ignore
    get_client = None  # type: ignore

class LLMClient(ABC):
    """
    Abstract Base Class for all LLM clients.
    Implementations must provide both:
      - generate(messages): full response
      - stream(messages): incremental chunks

    messages format:
    [
        {"role": "system", "content": "You are ..."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        # langfuse client is optional; only set when package and env are available
        self.langfuse_client = get_client() if get_client else None

    @abstractmethod
    def generate(self, messages: List[dict[str, Any]], tools: Optional[list] = None) -> list[dict]:
        """
        Send a list of messages to the model and return a full response.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: List[dict[str, Any]], tools: Optional[list] = None) -> Iterator[dict]:
        """
        Stream both reasoning tokens and final assistant output.

        Yields events shaped like: {"type": "reasoning", "token": "..."}
        or
        { "type": "content", "token": "..."}
        """
        raise NotImplementedError

    def observed_generate(self, messages: List[dict[str, Any]], tools: Optional[list] = None):
        """
        Optional langfuse-traced generate. Falls back to plain generate when langfuse
        is unavailable.
        """
        if observe is None:
            return self.generate(messages, tools=tools)

        @observe(name="llm-call", as_type="generation")
        def _call():
            return self.generate(messages, tools=tools)

        return _call()

    def observed_stream(self, messages: List[dict[str, Any]], tools: Optional[list] = None):
        """
        Optional langfuse-traced stream. Falls back to plain stream when langfuse
        is unavailable.
        """
        if observe is None:
            return self.stream(messages, tools=tools)

        @observe(name="llm-stream", as_type="generation")
        def _call():
            return list(self.stream(messages, tools=tools))

        return _call()
