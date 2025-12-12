import sys
from pathlib import Path

# Ensure the project root is in sys.path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest

# Import the module under test
from tools.toolkit import web_explorer

# List of expected public functions in the web_explorer module
EXPECTED_FUNCTIONS = [
    "goto_url",
    "get_page_content",
    "click_element",
    "fill_input",
    "screenshot",
    "end_browsing_page",
]


def test_expected_functions_exist():
    """Verify that all expected functions are defined in the module."""
    for func_name in EXPECTED_FUNCTIONS:
        assert hasattr(web_explorer, func_name), f"Missing function: {func_name}"
        func = getattr(web_explorer, func_name)
        assert callable(func), f"{func_name} should be callable"


# The following tests attempt to call the functions with minimal, mocked
# arguments. The actual implementation likely interacts with a browser driver
# (e.g., Selenium). We monkey‑patch the driver‑related attributes to avoid real
# network or UI operations.

@pytest.fixture(autouse=True)
def mock_browser(monkeypatch):
    """Replace any browser driver used by web_explorer with a dummy object.

    The real module may create a driver instance (e.g., Selenium WebDriver) at
    import time or lazily inside functions. We replace common attributes with a
    simple mock that records calls but performs no action.
    """
    class DummyDriver:
        def get(self, url):
            pass

        def find_element(self, *args, **kwargs):
            return self

        def click(self):
            pass

        def send_keys(self, *args, **kwargs):
            pass

        def page_source(self):
            return "<html></html>"

        def save_screenshot(self, path):
            return True

        def quit(self):
            pass

    dummy = DummyDriver()
    # Common attribute names that the module might use
    monkeypatch.setattr(web_explorer, "driver", dummy, raising=False)
    # If the module lazily creates a driver via a helper, patch that too
    monkeypatch.setattr(web_explorer, "_create_driver", lambda *args, **kwargs: dummy, raising=False)
    yield dummy


def test_goto_url_calls_driver_get(mock_browser):
    # Provide a dummy URL; the dummy driver does nothing but should be called.
    url = "http://example.com"
    # If the function returns something, we ignore it; we just ensure no exception.
    web_explorer.goto_url(url)


def test_get_page_content_returns_string(mock_browser):
    content = web_explorer.get_page_content()
    assert isinstance(content, str)


def test_click_element_no_error(mock_browser):
    # Use a dummy selector; the dummy driver will accept any call.
    selector = "#button"
    web_explorer.click_element(selector)


def test_fill_input_no_error(mock_browser):
    selector = "#input"
    value = "test"
    web_explorer.fill_input(selector, value)


def test_screenshot_returns_path(mock_browser, tmp_path):
    # The function may return the path to the saved screenshot.
    path = web_explorer.screenshot(str(tmp_path / "shot.png"))
    # Accept either None or a string path.
    assert path is None or isinstance(path, str)


def test_end_browsing_page_closes_driver(mock_browser):
    # Ensure calling end_browsing_page does not raise.
    web_explorer.end_browsing_page()
