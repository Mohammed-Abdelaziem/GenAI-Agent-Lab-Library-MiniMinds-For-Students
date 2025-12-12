"""
This is an Agent who given certain python files will write unit tests using pytest 
And will execute them and report result

Difference Between it And `00_raw_traced_unit_tester.py` no mlflow/langfuse 
"""
import json
from contextlib import nullcontext
from loguru import logger
from langfuse import observe, get_client
from tools.registry import ToolRegistry
import tools.toolkit.web_explorer as web_explorer_tools
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from llm.groq_client import GroqClient, LLMConfig
from llm.config import LLMProvider
from pathlib import Path


langfuse = get_client()

@observe(name="llm-call", as_type="generation")
def traced_client_generate(client, messages, tools):
    return client.generate(messages, tools=tools)


@observe(name="tool-call", as_type="tool")
def traced_tool_execution(registery, tool_call):
    try:
        func_name = tool_call["function"]["name"]
        args_raw = tool_call["function"].get("arguments", {}) or {}
        if isinstance(args_raw, str):
            func_inputs = json.loads(args_raw)
        else:
            func_inputs = args_raw

        func = registery.get(func_name)
        if func is None:
            raise ValueError(f"Tool {func_name} not found")
        func_results = func(**func_inputs)

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "name": func_name,
            "content": json.dumps(func_results),
        }
        return tool_message
    except Exception as error:
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "content": str(error),
        }
                
# ================ 1. Initalization ================
# 1.1 setup llm client
config = LLMConfig(
    max_tokens=5000,
    model_name="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 1.2 setup path
messages = [
    {
        "role": "system", "content": """
        You are a highly skilled QA Automation Agent with expertise in Python programming, unit testing (using Pytest), and modern GenAI tools. 
        Your role is to review, write, and execute comprehensive unit tests for the provided toolkit modules to ensure reliability and correctness. 
        Analyze each tool's behavior, suggest improvements if needed, and provide a clear, structured test report based on your findings.
        
        Output:
            - "finished": <boolean, indicate if the task is complete>
            - "message": <summary and coverage of tests>
        
        **Use only this tools:**
        {tools}
        you will be penalized if you use OTHER TOOLS
        """
    },
    {
        "role": "user", "content": """write unite tests for file: {files_under_test} code and run it ensure everything is okay then report it
        You are only allowed to change files in this directory {tests_output_directory_path}
        """ 
    }
]
# 1.3 create tool register and add tools/modules you need
registery = ToolRegistry()
registery.register(file_tools.write_file)
registery.register(file_tools.read_file)
registery.register(code_tools.run_pytest_tests)
registery.register(file_tools.list_directory_files)
registery.register(json_tools.json_is_valid)
registery.register(web_explorer_tools.goto_url)
registery.register(web_explorer_tools.get_page_content)
registery.register(web_explorer_tools.click_element)
registery.register(web_explorer_tools.fill_input)
registery.register(web_explorer_tools.screenshot)
registery.register(web_explorer_tools.end_browsing_page)

# 1.4 add tools to system_message use str.format method and  registery.to_string()
messages[0]["content"] = messages[0]["content"].format(tools=registery.to_string())

# 1.5 set `files_under_test` and `tests_output_directory_path` in user_message like .format
tests_output_directory_path = Path("tools/llm_tests")
tests_output_directory_path.mkdir(parents=True, exist_ok=True)
files_under_test = ["tools/toolkit/web_explorer.py"]
messages[1]["content"] = messages[1]["content"].format(
    files_under_test=files_under_test,
    tests_output_directory_path=tests_output_directory_path,
)

# 1.6 set root span with root_span = langfuse.start_span(name , metadata)
if langfuse:
    root_span = langfuse.start_span(
        name="raw-traced-unit-tester",
        metadata={
            "files_under_test": files_under_test,
            "tests_output_directory_path": str(tests_output_directory_path),
        },
    )
else:  # fallback if langfuse not configured
    root_span = nullcontext()

# ================ 2. Starts Iterations ================
max_iterations = 20
iteration = 0
while True:
    iteration += 1
    logger.info(f"Iteration {iteration}")
    with root_span.start_as_current_observation(as_type="span", name=f"iteration-{iteration}"):
        response = traced_client_generate(client, messages, tools=registery.to_client_tools(config.provider))[0]
        messages.append(
            {
                "role": response.get("role", "ai"),
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls"),
            }
        )
        logger.info(json.dumps(messages[-1], indent=2))

        # Stop when finished flag found or max iterations reached
        content = response.get("content") or ""
        finished_flag = False
        try:
            parsed = json.loads(content)
            finished_flag = isinstance(parsed, dict) and parsed.get("finished") is True
        except Exception:
            finished_flag = False

        if finished_flag or iteration >= max_iterations:
            break
        
        # 2.4 execute any function execution inside `tool_calls`
        tool_calls = response.get("tool_calls", []) or []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                continue

            tool_message = traced_tool_execution(registery, tool_call)
            messages.append(tool_message)
            logger.info(f"tool response {json.dumps(tool_message, indent=2)}")

if hasattr(root_span, "end"):
    root_span.end()
