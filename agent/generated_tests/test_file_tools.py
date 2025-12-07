import sys
import pathlib
# Add project root to sys.path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import os
import shutil
from pathlib import Path

import pytest

# Import the functions to test
from tools.toolkit.builtin.file_tools import (
    list_directory_files,
    read_file,
    write_file,
    create_folder,
    remove_folder,
    remove_file,
)


def test_write_and_read_file(tmp_path: Path):
    # Define a file path
    file_path = tmp_path / "test.txt"
    content = "Hello, world!"

    # Write content
    write_result = write_file(str(file_path), content)
    assert write_result["success"] is True
    assert write_result["result"] is True

    # Read back
    read_result = read_file(str(file_path))
    assert read_result["success"] is True
    assert read_result["result"] == content

    # Try reading non‑existent file
    missing = tmp_path / "missing.txt"
    read_missing = read_file(str(missing))
    assert read_missing["success"] is False
    assert "File not found" in read_missing["error"]


def test_create_and_remove_folder(tmp_path: Path):
    folder = tmp_path / "new_folder"
    # Create folder
    create_res = create_folder(str(folder))
    assert create_res["success"] is True
    assert folder.is_dir()

    # Creating again should fail
    create_again = create_folder(str(folder))
    assert create_again["success"] is False
    assert "already exists" in create_again["error"].lower()

    # Remove folder
    remove_res = remove_folder(str(folder))
    assert remove_res["success"] is True
    assert not folder.exists()

    # Removing non‑existent folder
    remove_missing = remove_folder(str(folder))
    assert remove_missing["success"] is False
    assert "not found" in remove_missing["error"].lower()


def test_remove_file(tmp_path: Path):
    # Create a temporary file
    file_path = tmp_path / "temp.txt"
    file_path.write_text("temp")
    assert file_path.is_file()

    # Remove it
    remove_res = remove_file(str(file_path))
    assert remove_res["success"] is True
    assert not file_path.exists()

    # Removing again should error
    remove_again = remove_file(str(file_path))
    assert remove_again["success"] is False
    assert "not found" in remove_again["error"].lower()


def test_list_directory_files_depth(tmp_path: Path):
    # Create a nested structure:
    # tmp_path/
    #   a.txt
    #   sub1/
    #       b.txt
    #       sub2/
    #           c.txt
    (tmp_path / "a.txt").write_text("a")
    sub1 = tmp_path / "sub1"
    sub1.mkdir()
    (sub1 / "b.txt").write_text("b")
    sub2 = sub1 / "sub2"
    sub2.mkdir()
    (sub2 / "c.txt").write_text("c")

    # Depth 0 should only list the root directory
    res0 = list_directory_files(str(tmp_path), depth=0)
    assert res0["success"] is True
    root_key = str(tmp_path)
    assert root_key in res0["result"]
    # Should contain a.txt and sub1
    root_contents = set(res0["result"][root_key])
    assert root_contents == {"a.txt", "sub1"}
    # No deeper entries
    assert len(res0["result"]) == 1

    # Depth 1 should include sub1 contents as well
    res1 = list_directory_files(str(tmp_path), depth=1)
    assert res1["success"] is True
    assert root_key in res1["result"]
    sub1_key = str(sub1)
    assert sub1_key in res1["result"]
    # Verify sub1 contents
    sub1_contents = set(res1["result"][sub1_key])
    assert sub1_contents == {"b.txt", "sub2"}
    # No sub2 entry yet
    assert str(sub2) not in res1["result"]

    # Depth 2 should include sub2
    res2 = list_directory_files(str(tmp_path), depth=2)
    assert res2["success"] is True
    assert str(sub2) in res2["result"]
    sub2_contents = set(res2["result"][str(sub2)])
    assert sub2_contents == {"c.txt"}

    # Non‑existent path
    missing = tmp_path / "no_such"
    res_missing = list_directory_files(str(missing), depth=1)
    assert res_missing["success"] is False
    assert "Path not found" in res_missing["error"]
