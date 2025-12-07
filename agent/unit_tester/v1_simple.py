from ..base import Agent, BaseAgentState, LLMClient, ToolRegistry
from tools.toolkit.builtin import code_tools, file_tools, json_tools
from pathlib import Path
import json
from loguru import logger
from typing import Optional


class SimpleUnitTesterAgent(Agent):
    def __init__(self, llm: LLMClient,  max_iterations: int = 100):
        # create tool registry with only the tools needed to write/run tests
        tool_registry = ToolRegistry()
        tool_registry.register(file_tools.write_file)
        tool_registry.register(file_tools.read_file)
        tool_registry.register(code_tools.run_pytest_tests)
        # optional: allow listing files to locate target
        tool_registry.register(file_tools.list_directory_files)
        # json_is_valid can help validate model outputs
        tool_registry.register(json_tools.json_is_valid)

        super().__init__(llm, tool_registry, max_iterations)
        # initialize state with system prompt
        prompt_path = Path("prompts/unit_tester_v1.txt")
        system_prompt_template = prompt_path.read_text(encoding="utf-8")
        system_prompt = system_prompt_template.format(
            tools=self.tool_registry.to_string()
        )
        self.inital_state.add_message(role="system", content=system_prompt)

    
    def start_point(self, user_query) -> BaseAgentState:
        """ Start Point of State for example start of user query or anything """
        state = self.inital_state
        state.add_message(role="user", content=user_query)

        return state
    
    def run(self, state: BaseAgentState) -> BaseAgentState:
        # 1) Call LLM
        response = self.llm_generate(state)[0]

        # 2) Add assistant response
        state.add_message(role=response.get("role", "ai"), content=response.get("content", ""), tool_calls=response.get("tool_calls"))

        pytest_passed = False
        test_files_written = set()

        # 4) Execute tool calls if any
        tool_calls = response.get("tool_calls", []) or []
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                continue
            func_name = tool_call["function"]["name"]

            # parse arguments once
            args_raw = tool_call["function"].get("arguments", {}) or {}
            if isinstance(args_raw, str):
                try:
                    func_inputs = json.loads(args_raw)
                except Exception:
                    func_inputs = {}
            else:
                func_inputs = args_raw

            # track when we actually write a test file into tools/llm_tests
            if func_name == "write_file":
                path_arg = func_inputs.get("file_path")
                if path_arg and str(path_arg).startswith("tools/llm_tests"):
                    test_files_written.add(str(path_arg))

            # skip premature pytest runs before any test file exists
            if func_name == "run_pytest_tests" and not test_files_written:
                logger.debug("Skipping run_pytest_tests until a test file is written")
                continue

            # call tool using already parsed inputs
            tool_call_copy = dict(tool_call)
            tool_call_copy["function"] = dict(tool_call["function"])
            tool_call_copy["function"]["arguments"] = func_inputs
            tool_result = self.call_tool(tool_call_copy)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "name": func_name,
                "content": json.dumps(tool_result),
            }
            state.messages.append(tool_message)
            logger.info(json.dumps(tool_message, indent=2))

            if func_name == "run_pytest_tests" and isinstance(tool_result, dict):
                result_text = str(tool_result.get("result", ""))
                collected_ok = False
                if "collected" in result_text:
                    # crude parse to ensure tests actually ran
                    try:
                        after_collected = result_text.split("collected", 1)[1]
                        num = after_collected.split("items")[0].strip().split()[-1]
                        collected_ok = int(num) > 0
                    except Exception:
                        collected_ok = False
                if (
                    tool_result.get("success")
                    and "FAIL" not in result_text
                    and "ERROR" not in result_text
                    and "no tests ran" not in result_text.lower()
                    and collected_ok
                ):
                    pytest_passed = True

        # If we have written tests but haven't run pytest yet, force a pytest call
        if test_files_written and not pytest_passed:
            pytest_call = {
                "type": "function",
                "id": "forced-pytest",
                "function": {"name": "run_pytest_tests", "arguments": {"directory": "tools/llm_tests"}},
            }
            tool_result = self.call_tool(pytest_call)
            tool_message = {
                "role": "tool",
                "tool_call_id": pytest_call.get("id"),
                "name": "run_pytest_tests",
                "content": json.dumps(tool_result),
            }
            state.messages.append(tool_message)
            logger.info(json.dumps(tool_message, indent=2))

            if isinstance(tool_result, dict):
                result_text = str(tool_result.get("result", ""))
                collected_ok = False
                if "collected" in result_text:
                    try:
                        after_collected = result_text.split("collected", 1)[1]
                        num = after_collected.split("items")[0].strip().split()[-1]
                        collected_ok = int(num) > 0
                    except Exception:
                        collected_ok = False
                if (
                    tool_result.get("success")
                    and "FAIL" not in result_text
                    and "ERROR" not in result_text
                    and "no tests ran" not in result_text.lower()
                    and collected_ok
                ):
                    pytest_passed = True
                else:
                    # add quick hint message to state to steer next turn
                    state.add_message(
                        role="ai",
                        content="Pytest failed or found no tests; fix imports (tools path) or failing tests, then rerun."
                    )

        # 3) Set Stop condition (only when pytest passed)
        if pytest_passed:
            finish_msg = {
                "finished": True,
                "message": "Pytest run completed successfully."
            }
            state.add_message(role="ai", content=json.dumps(finish_msg))
            state.is_finished = True

        # 5) return state
        return state
