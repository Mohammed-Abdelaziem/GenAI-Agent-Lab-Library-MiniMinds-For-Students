from pathlib import Path

from dotenv import load_dotenv

from agent.unit_tester.v2_scratchpad import ScratchpadUnitTesterAgent
from llm.groq_client import GroqClient, LLMConfig


def main():
    load_dotenv()

    config = LLMConfig(
        # Smaller cap to avoid hitting TPM limits on long prompts
        max_tokens=3000,
        model_name="openai/gpt-oss-120b",
        reasoning_effort="medium",
        temperature=0.3,
        top_p=0.8,
    )
    client = GroqClient(config)

    tests_output_directory_path = Path("tools/llm_tests")
    tests_output_directory_path.mkdir(exist_ok=True, parents=True)

    files_under_test = ["tools/toolkit/web_explorer.py"]
    user_query = f"""
    STOP calling read_file repeatedly. You must now write tests to tools/llm_tests/test_web_explorer.py and run_pytest_tests in tools/llm_tests.
    Fix tests to match the actual module:
    - Do not patch requests.get unless you add an import; avoid patching non-existent attributes.
    - Assert on functions that exist in tools/toolkit/web_explorer.py (goto_url, get_page_content, click_element, fill_input, screenshot, end_browsing_page), not on missing symbols.
    Write pytest unit tests for: {files_under_test}
    - Write tests to a single file under {str(tests_output_directory_path)} (e.g., test_web_explorer.py).
    - You may mock external/browser/network interactions as needed; focus on verifying our code paths (no real network/HTTP calls).
    - Use pytest functions (no unittest.main). Add sys.path.append(str(Path(__file__).resolve().parents[2])) so imports like `from tools.toolkit import web_explorer` work.
    - After writing the test file, run pytest in {str(tests_output_directory_path)} and report results. If imports fail or no tests collected, fix and retry.
    - Do not touch files outside {str(tests_output_directory_path)}.
    - Do not call list_directory_files more than once. Next actions after listing: read_file tools/toolkit/web_explorer.py, write tests into tools/llm_tests/test_web_explorer.py, then run_pytest_tests on tools/llm_tests.
    - Do not call read_file on the same path more than once. You already read tools/toolkit/web_explorer.py; move on to writing tests and running pytest in tools/llm_tests.
    - Stop calling read_file again. Now write tests to tools/llm_tests/test_web_explorer.py and then run_pytest_tests on tools/llm_tests. Do not continue until you’ve written the test file.
    """.strip()

    agent = ScratchpadUnitTesterAgent(client, max_iterations=10)
    state = agent.iterate(user_query=user_query)

    print("is_finished:", state.is_finished)
    print("scratchpad:", state.scratchpad)
    print("messages (tail):", state.messages[-3:])


if __name__ == "__main__":
    main()
