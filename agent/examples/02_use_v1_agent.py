from agent.unit_tester.v1_simple import SimpleUnitTesterAgent
from llm.groq_client import GroqClient, LLMConfig
from pathlib import Path
from dotenv import load_dotenv

# load env for langfuse / groq keys
load_dotenv()

# 1. llm client (using Groq client already available)
config = LLMConfig(
    max_tokens=5000,
    model_name="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=1.0,
    top_p=1
)
client = GroqClient(config)

# 2. user query
tests_output_directory_path = Path("tools/llm_tests")
tests_output_directory_path.mkdir(exist_ok=True, parents=True)

files_under_test = ["tools/toolkit/web_explorer.py"]
user_query = f"""
Write pytest unit tests for: {files_under_test}
- Write tests to a single file under {str(tests_output_directory_path)} (e.g., test_web_explorer.py).
- You may mock external/browser/network interactions as needed; focus on verifying our code paths (no real network/HTTP calls).
- Use pytest functions (no unittest.main). Add sys.path.append(str(Path(__file__).resolve().parents[2])) so imports like `from tools.toolkit import web_explorer` work.
- After writing the test file, run pytest in {str(tests_output_directory_path)} and report results. If imports fail or no tests collected, fix and retry.
- Do not touch files outside {str(tests_output_directory_path)}.
"""
        
# 3. run agent

agent = SimpleUnitTesterAgent(client, max_iterations=6)

state = agent.iterate(user_query = user_query)

print(state)
