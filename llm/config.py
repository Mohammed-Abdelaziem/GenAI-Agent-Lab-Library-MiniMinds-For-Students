import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    
    provider: LLMProvider = Field(default=LLMProvider.GROQ, description="The LLM provider to use")
    
    base_url: Optional[str] = Field( default=None, description="API Base URL (only needed for custom endpoints)")

    model_name: str = Field( default="llama-3.1-70b-versatile", description="model name")
    
    # Common configuration
    # TODO: add temperature field with validations (0.0 <= temperature <= 2.0) and make default = 0.7
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values = more creative responses."
    )    
    # TODO: add top_p field with validations (0.0 <= top_p <= 1.0) and make default = 0.7
    top_p: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling. Controls diversity by limiting cumulative probability."
    )    
    # TODO: add max_tokens field
    max_tokens: Optional[int] = Field(
        default=2048,
        description="Maximum number of tokens to generate in the output."
    )    
    # TODO: search on reasoning_effort and what it do (write description to this field)
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Controls the intensity of structured reasoning for models that support it "
            "(e.g., Groq 'reasoning' mode). Examples: 'low', 'medium', 'high'. "
            "Not all models or providers use this field."
        )
    )