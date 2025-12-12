import json
from pathlib import Path
from typing import List
from loguru import logger

from ..base import Agent, ScratchpadAgentState, prune_messages
from llm.base import LLMClient
from tools.registry import ToolRegistry
from tools.toolkit.builtin import code_tools, file_tools, json_tools


class ScratchpadUnitTesterAgent(Agent):
    """
    Unit tester agent v2:
    - Keeps a scratchpad summary after each tool call.
    - Prunes older tool/assistant messages to avoid context bloat.
    """

    def __init__(self, llm: LLMClient, max_iterations: int = 100):
        tool_registry = ToolRegistry()
        tool_registry.register(file_tools.write_file)
        tool_registry.register(file_tools.read_file)
        tool_registry.register(code_tools.run_pytest_tests)
        # tool_registry.register(file_tools.list_directory_files)
        tool_registry.register(json_tools.json_is_valid)

        super().__init__(llm, tool_registry, max_iterations)

        prompt_path = Path("prompts/unit_tester_v2.txt")
        system_prompt_template = prompt_path.read_text(encoding="utf-8")
        system_prompt = system_prompt_template.format(
            tools=self.tool_registry.to_string()
        )

        self.initial_state = ScratchpadAgentState(
            messages=[{"role": "system", "content": system_prompt}],
            scratchpad=[],
            test_files_written=set(),
            pytest_passed=False,
        )

    def start_point(self, user_query) -> ScratchpadAgentState:
        state = self.initial_state.model_copy(deep=True)
        state.add_message(role="user", content=user_query)
        return state

    def run(self, state: ScratchpadAgentState) -> ScratchpadAgentState:
        response = self.llm_generate(state)[0]
        state.messages.append(
            {
                "role": response.get("role", "ai"),
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls"),
            }
        )

        tool_calls = response.get("tool_calls") or []
        scratchpad_entries: List[str] = []
        test_files_written = set(state.test_files_written)
        pytest_passed = state.pytest_passed

        def summarize_tool(func_name: str, tool_result, func_inputs: dict) -> str:
            if func_name == "list_directory_files":
                result = tool_result.get("result", {})
                if isinstance(result, dict) and "result" in result:
                    result = result.get("result", {})
                summary = {}
                if isinstance(result, dict):
                    for dir_path, items in result.items():
                        if not isinstance(items, list):
                            continue
                        filtered = [
                            i
                            for i in items
                            if not str(i).startswith((".git", ".venv", "__pycache__"))
                        ]
                        summary[dir_path] = {
                            "total": len(filtered),
                            "sample": filtered[:6],
                        }
                return f"{func_name} summary: {json.dumps(summary)[:350]}"
            return f"{func_name}: {str(tool_result)[:350]}"

        # Execute tool calls
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                continue
            func_name = tool_call["function"]["name"]
            args_raw = tool_call["function"].get("arguments", {}) or {}
            if isinstance(args_raw, str):
                try:
                    func_inputs = json.loads(args_raw)
                except Exception:
                    func_inputs = {}
            else:
                func_inputs = args_raw

            if func_name == "list_directory_files":
                path_normalized = func_inputs.get("path", "") or "."
                if path_normalized in {"./"}:
                    path_normalized = "."
                func_inputs["path"] = path_normalized

                if state.dir_listings_executed >= 1:
                    skip_msg = (
                        "list_directory_files disabled after the initial call; "
                        "proceed to read_file tools/toolkit/web_explorer.py and write tests."
                    )
                    scratchpad_entries.append(skip_msg)
                    state.add_message(
                        role="assistant",
                        content=(
                            "Stop listing directories. Next: read_file tools/toolkit/web_explorer.py, "
                            "then write tests into tools/llm_tests/test_web_explorer.py and run pytest there."
                        ),
                    )
                    continue
                depth_val = func_inputs.get("depth", 2)
                try:
                    func_inputs["depth"] = max(1, min(int(depth_val), 2))
                except Exception:
                    func_inputs["depth"] = 2

                signature = json.dumps(
                    {"path": path_normalized, "depth": func_inputs["depth"]},
                    sort_keys=True,
                )
                if state.target_module_read or state.dir_listings_executed >= 2:
                    skip_msg = (
                        "directory listings disabled after initial exploration; "
                        "read target module and proceed to tests."
                    )
                    scratchpad_entries.append(skip_msg)
                    state.add_message(
                        role="assistant",
                        content=(
                            "Next action: read_file tools/toolkit/web_explorer.py, "
                            "draft tests into tools/llm_tests/test_web_explorer.py, "
                            "then run pytest in tools/llm_tests."
                        ),
                    )
                    continue
                if signature in state.recent_dir_signatures[-3:]:
                    skip_msg = f"skipped duplicate list_directory_files for {signature}"
                    scratchpad_entries.append(skip_msg)
                    continue
                if state.dir_listings_executed >= 2:
                    skip_msg = (
                        "max directory listings reached; move to reading target module "
                        "tools/toolkit/web_explorer.py then write tests in tools/llm_tests."
                    )
                    scratchpad_entries.append(skip_msg)
                    state.add_message(
                        role="assistant",
                        content=(
                            "Next action: read_file tools/toolkit/web_explorer.py, "
                            "plan tests, write a single pytest file under tools/llm_tests/, "
                            "then run pytest in tools/llm_tests."
                        ),
                    )
                    continue

            if func_name == "write_file":
                path_arg = func_inputs.get("file_path")
                if path_arg and str(path_arg).startswith("tools/llm_tests"):
                    test_files_written.add(str(path_arg))

            if func_name == "read_file":
                path_arg = func_inputs.get("file_path")
                if path_arg == "tools/toolkit/web_explorer.py":
                    state.target_module_read = True
                if path_arg and path_arg in state.read_files_seen:
                    skip_msg = (
                        f"skipped duplicate read_file for {path_arg}; proceed to write tests "
                        "into tools/llm_tests/test_web_explorer.py and run pytest there."
                    )
                    scratchpad_entries.append(skip_msg)
                    # Return a tool-style error to make the LLM advance
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": func_name,
                        "content": json.dumps(
                            {
                                "success": False,
                                "error": "read_file already executed for this path; write tests now and run pytest.",
                            }
                        ),
                    }
                    state.messages.append(tool_message)
                    logger.info(json.dumps(tool_message, indent=2))
                    state.add_message(
                        role="assistant",
                        content=(
                            "Do not re-read the same file. Move on: write tests into "
                            "tools/llm_tests/test_web_explorer.py, then run pytest in tools/llm_tests."
                        ),
                    )
                    continue

            if func_name == "run_pytest_tests" and not test_files_written:
                logger.debug("Skipping run_pytest_tests until a test file is written")
                continue

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

            if func_name == "list_directory_files":
                state.recent_dir_signatures.append(signature)
                state.recent_dir_signatures = state.recent_dir_signatures[-5:]
                state.dir_listings_executed += 1
            if func_name == "read_file":
                path_arg = func_inputs.get("file_path")
                if path_arg:
                    state.read_files_seen.add(path_arg)
            scratchpad_entries.append(summarize_tool(func_name, tool_result, func_inputs))

            if func_name == "run_pytest_tests" and isinstance(tool_result, dict):
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

        # Force pytest once a test file exists and no passing run yet
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
            scratchpad_entries.append(f"forced run_pytest_tests: {str(tool_result)[:500]}")

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
                    state.add_message(
                        role="ai",
                        content="Pytest failed or found no tests; fix imports (tools path) or failing tests, then rerun."
                    )

        # Update scratchpad and prune history to keep context small
        scratchpad_payload = None
        if scratchpad_entries:
            state.scratchpad.extend(scratchpad_entries)
            scratchpad_payload = {
                "iteration": state.iteration,
                "entries": scratchpad_entries,
                "written_tests": sorted(test_files_written),
                "pytest_passed": pytest_passed,
            }
            state.messages.append(
                {
                    "role": "assistant",
                    "content": f"<scratchpad>{json.dumps(scratchpad_payload)}</scratchpad>",
                }
            )
        state.messages = prune_messages(state.messages, drop_tools=True, last_n=4)

        # Stop condition
        if pytest_passed:
            finish_msg = {
                "finished": True,
                "message": "Pytest run completed successfully.",
                "scratchpad": state.scratchpad,
            }
            state.add_message(role="ai", content=json.dumps(finish_msg))
            state.is_finished = True

        # Persist tracking flags
        state.test_files_written = test_files_written
        state.pytest_passed = pytest_passed
        return state
