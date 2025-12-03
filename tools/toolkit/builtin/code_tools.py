from tools.decorator import tool
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
        # TODO:
        p = Path(directory)
        if not p.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        proc = subprocess.run(
            ["pytest", str(p)],
            capture_output=True,
            text=True
        )
        output = proc.stdout + "\n" + proc.stderr

        return {"success": True, "result": output.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}
