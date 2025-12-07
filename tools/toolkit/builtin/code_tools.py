from tools.decorator import tool
import os
from pathlib import Path
import subprocess

@tool()
def run_python_file(file_path: str) -> dict:
    """
    Run a Python file and return its stdout and stderr.
    Returns output as a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        proc = subprocess.run(
            ["python", str(p)],
            capture_output=True,
            text=True
        )

        output = proc.stdout + "\n" + proc.stderr

        return {"success": True, "result": output.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def run_pytest_tests(directory: str = ".") -> dict:
    """
    Run pytest in the given directory and return its output.
    Returns output as a dictionary with success/error status and result/message.
    """
    try:
        p = Path(directory)
        if not p.exists() or not p.is_dir():
            return {"success": False, "error": f"Not a directory: {directory}"}

        # Ensure imports from repo root work during pytest execution
        env = dict(**os.environ)
        repo_root = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = str(repo_root)

        proc = subprocess.run(
            ["pytest", "."],
            capture_output=True,
            text=True,
            cwd=str(p),
            env=env,
        )
        output = proc.stdout + "\n" + proc.stderr

        return {"success": proc.returncode == 0, "result": output.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}
