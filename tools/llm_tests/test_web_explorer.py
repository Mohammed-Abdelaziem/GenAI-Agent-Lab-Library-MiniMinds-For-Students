import sys
from pathlib import Path

# Ensure the toolkit package is importable
sys.path.append(str(Path(__file__).resolve().parents[2]))

import builtins
import types
import pytest

# Import the module under test
from tools.toolkit import web_explorer

# Helper mock classes
class MockElement:
    def __init__(self):
        self.clicked = False
        self.sent_keys = None
        self.screenshot_data = b"pngdata"

    def click(self):
        self.clicked = True

    def send_keys(self, text):
        self.sent_keys = text

    def screenshot_as_png(self):
        return self.screenshot_data

class MockDriver:
    def __init__(self):
        self.last_url = None
        self.page_source = "<html></html>"
        self.quitted = False
        self.elements = {}
        self.screenshot_taken = False

    def get(self, url):
        self.last_url = url

    def find_element(self, by, value):
        # Simplify: ignore 'by', just use value as key
        if value not in self.elements:
            self.elements[value] = MockElement()
        return self.elements[value]

    def get_screenshot_as_png(self):
        self.screenshot_taken = True
        return b"pngbytes"

    def quit(self):
        self.quitted = True

# Fixture to inject a mock driver into the module
@pytest.fixture(autouse=True)
def mock_driver(monkeypatch):
    driver = MockDriver()
    # The module may store the driver in a private variable; we attempt common names
    # If the module defines a global _driver, replace it; otherwise set an attribute
    if hasattr(web_explorer, "_driver"):
        monkeypatch.setattr(web_explorer, "_driver", driver, raising=False)
    else:
        monkeypatch.setattr(web_explorer, "driver", driver, raising=False)
    return driver

def test_goto_url_calls_driver_get(mock_driver):
    url = "https://example.com"
    # Call the function under test
    web_explorer.goto_url(url)
    # Verify the driver received the URL
    assert mock_driver.last_url == url

def test_get_page_content_returns_source(mock_driver):
    # Set a custom page source to verify return value
    mock_driver.page_source = "<html><body>Hello</body></html>"
    content = web_explorer.get_page_content()
    assert content == mock_driver.page_source

def test_click_element_finds_and_clicks(mock_driver):
    selector = "#button"
    # Ensure element does not exist before call
    assert selector not in mock_driver.elements
    web_explorer.click_element(selector)
    # After call, element should exist and be clicked
    element = mock_driver.elements.get(selector)
    assert element is not None
    assert element.clicked is True

def test_fill_input_sends_keys(mock_driver):
    selector = "#input"
    text = "hello world"
    web_explorer.fill_input(selector, text)
    element = mock_driver.elements.get(selector)
    assert element is not None
    assert element.sent_keys == text

def test_screenshot_calls_driver_and_returns_bytes(mock_driver):
    # The screenshot function may accept a path; we ignore it for the test
    result = web_explorer.screenshot()
    # Verify driver method was called and bytes were returned
    assert mock_driver.screenshot_taken is True
    assert isinstance(result, (bytes, bytearray))
    assert result == b"pngbytes"

def test_end_browsing_page_quits_driver(mock_driver):
    web_explorer.end_browsing_page()
    assert mock_driver.quitted is True
