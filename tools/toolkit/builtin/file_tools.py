from tools.decorator import tool
from pathlib import Path
import shutil

@tool()
def list_directory_files(path: str = ".", depth: int = 1) -> dict:
    """
    List files and directories in the given path up to a certain depth using pathlib.
    Returns a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        base = Path(path)
        
        if not base.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        result = {}

        def walk(p: Path, d: int):
            if d < 0:
                return
            result[str(p)] = [f.name for f in p.iterdir()]
            for f in p.iterdir():
                if f.is_dir():
                    walk(f, d - 1)

        walk(base, depth)
        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def read_file(file_path: str) -> dict:
    """
    Read the content of a file.
    Returns a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        content = p.read_text(encoding="utf-8")
        return {"success": True, "result": content}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def write_file(file_path: str, content: str) -> dict:
    """
    Write content to a file.
    Returns a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(file_path)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "result": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def create_folder(folder_path: str) -> dict:
    """
    Create a new folder (directory) at the specified path.
    Returns a dictionary with success/error status and result/message.
    """
    # TODO:
    try:
        p = Path(folder_path)
        p.mkdir(parents=True, exist_ok=False)
        return {"success": True, "result": True}
    except FileExistsError:
        return {"success": False, "error": "Folder already exists"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def remove_folder(folder_path: str) -> dict:
    """
    Remove a folder (directory) and all its contents at the specified path.
    Returns a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(folder_path)
        if not p.is_dir():
            return {"success": False, "error": f"Folder not found: {folder_path}"}
        shutil.rmtree(p)
        return {"success": True, "result": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def remove_file(file_path: str) -> dict:
    """
    Remove a file at the specified path.
    Returns a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(file_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        p.unlink()
        return {"success": True, "result": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
