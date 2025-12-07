import sys
from pathlib import Path

# Add repo root to sys.path for imports like "from tools.toolkit ..."
repo_root = Path(__file__).resolve().parents[2]
root_str = str(repo_root)
if root_str not in sys.path:
    sys.path.append(root_str)
