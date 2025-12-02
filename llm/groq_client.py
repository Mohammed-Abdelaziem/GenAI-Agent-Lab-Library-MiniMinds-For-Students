import os
from typing import Iterator, List
from dotenv import load_dotenv
from groq import Groq
from .base import LLMClient
from .config import LLMConfig
from messages.base import Message
from messages.human import HumanMessage
from messages.ai import AIMessage
from messages.thinking import ThinkingMessage
from messages.tool import ToolMessage

# TODO 1: load dotenv
load_dotenv()

class GroqClient(LLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # TODO 2: create groq client and set api_key from .env
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    def generate(self, messages: List[Message], tools=None) -> List[Message]:
        formatted = self.format_messages(messages)

        # TODO: write description for Returend Fields 
        """ Main Returned Dict Fields
        - content: The LLM's generated text
        - role: 'assistant' (mapped internally by Groq)
        - reasoning: optional chain-of-thought (only for reasoning models)
        """
        # TODO 3: call `client.chat.completions.create` with configurations in self.config
        # TODO 3: search difference between max_tokens and max_compeletion_tokens:
        # ANS: max_tokens -> OpenAI style: sets the whole limit for output tokens.
        # max_completion_tokens -> Groq style: specifically for completion output only.
        # TODO 3: now you can pass tools=tools but search about format later when move to tools sections
        response = self.client.chat.completions.create(
        model=self.config.model_name,
        messages=formatted,
        temperature=self.config.temperature,
        top_p=self.config.top_p,
        max_tokens=self.config.max_tokens,
        tools=tools,
        )
        ai_text = response.choices[0].message["content"]
        return [AIMessage(content=ai_text)]
    
    def stream(self, messages: List[Message], tools=None):
        formatted = self.format_messages(messages)

        # TODO 3: call `client.chat.completions.create` with stream options configurations in self.config
        stream = self.client.chat.completions.create(
        model=self.config.model_name,
        messages=formatted,
        temperature=self.config.temperature,
        top_p=self.config.top_p,
        max_tokens=self.config.max_tokens,
        stream=True,
        tools=tools,
        )
        for chunk in stream:
            if "content" in chunk.choices[0].delta:
                yield AIMessage(content=chunk.choices[0].delta["content"])                
    def format_messages(self, messages: List[Message]):
        formatted = []
        for msg in messages:
            if msg.role == "human":
                formatted.append({"role": "user", "content": msg.content})
            elif msg.role == "ai":
                formatted.append({"role": "assistant", "content": msg.content})
            elif msg.role == "thinking":
                formatted.append({"role": "reasoning", "content": msg.content})
            elif msg.role == "tool":
                formatted.append({"role": "tool", "content": msg.content, "name": msg.tool_name})
        return formatted
if __name__ == "__main__":
    #TODO: initlaize configuraiton with reasoning model -- search for groq reasoning models 
    config = LLMConfig(
        provider="groq",
        model_name="llama3-groq-70b-thinking",
        temperature=0.7,
        top_p=0.7,
        max_tokens=2000
    )
    client = GroqClient(config)
    
    #TODO: write messages with (1. system prompt on how the model is QA engineer and know python, playwright etc... provide in course) 
    #TODO: (2. Ask model to "write a plan to build a software autonomus like cursor but for testing") 
    messages = [
    HumanMessage(content="You are a QA engineer expert in Python, Playwright, and testing automation."),
    ]
    
    #TODO: test client.generate
    print("/////////// generate() ////////////")
    response = client.generate(messages)
    print(response)
    #TODO: test client.stream and mention what's difference and why we need it?
    # YOUR_ANSWER: 
    # generate() → waits for the full response then returns it all at once.
    # stream()   → returns the answer in small chunks (tokens) as the model is generating.
    print("/////////// stream() ////////////")
    for chunk in client.stream(messages):
        print(chunk)
    
    #TODO add the new answer to messages -> create multi-turn conversation (with same system message from above)
        # user: your name is CHATTAH tester
        # assisstant: ...
        # user: tell me what's your name and what are language you expert in it ?
    messages.append(HumanMessage(content="your name is CHATTAH tester"))

    # TODO: first turn -> get answer -> print answer -> append answer to messages "i.e state"
    answer1 = client.generate(messages)
    print(answer1)
    messages.extend(answer1)    

    # TODO: second turn -> get answer -> print answer -> append answer to messages "i.e state"
    messages.append(HumanMessage(content="tell me what's your name and what languages you are expert in?"))
    answer2 = client.generate(messages)
    print(answer2)
    messages.extend(answer2)

