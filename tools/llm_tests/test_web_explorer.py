import sys
from pathlib import Path
import base64

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pytest
from unittest.mock import MagicMock, patch

from tools.toolkit import web_explorer

# Helper mock page
class MockPage:
    def __init__(self):
        self.title_val = "Mock Title"
        self.url = "http://example.com"
        self._content = "<html><body>Mock Content</body></html>"
        self.screenshot_data = b"pngbytes"

    # Methods used in goto_url
    def goto(self, url, wait_until=None):
        self.url = url
        class Response:
            status = 200
        return Response()
    def title(self):
        return self.title_val

    # Methods for content
    def content(self):
        return self._content

    # locator handling
    def locator(self, selector):
        # Return a mock locator object with inner_text / inner_html / click / fill
        loc = MagicMock(name=f"Locator({selector})")
        if selector == "body":
            loc.inner_text.return_value = "Body Text"
        if selector == "html":
            # Provide inner_html method
            loc.inner_html.return_value = self._content
        # default click/fill behavior
        loc.click = MagicMock()
        loc.fill = MagicMock()
        # Provide _html attribute for fallback path
        loc._html = self._content
        return loc

    # get_by_* methods
    def get_by_text(self, text, exact=False):
        return MagicMock(name="get_by_text", click=MagicMock())
    def get_by_role(self, role, name=None):
        return MagicMock(name="get_by_role", click=MagicMock())

    # wait_for_load_state
    def wait_for_load_state(self, state, timeout=None):
        self.waited_state = (state, timeout)

    # screenshot
    def screenshot(self, full_page=False):
        return self.screenshot_data

# Mock close_page function

def mock_close_page(session_id):
    mock_close_page.called_with = session_id

@pytest.fixture(autouse=True)
def patch_browser_manager():
    with patch('tools.toolkit.web_explorer.get_page', return_value=MockPage()) as _g, \
         patch('tools.toolkit.web_explorer.close_page', side_effect=mock_close_page) as _c:
        yield

def test_goto_url_success():
    result = web_explorer.goto_url('http://test.com', session_id='default')
    assert "Navigated to" in result
    assert "Mock Title" in result
    assert "http://test.com" in result
    assert "HTTP Status: 200" in result

def test_goto_url_exception(monkeypatch):
    # Make page.goto raise
    class BadPage(MockPage):
        def goto(self, url, wait_until=None):
            raise RuntimeError("boom")
    monkeypatch.setattr(web_explorer, 'get_page', lambda sid: BadPage())
    result = web_explorer.goto_url('http://fail.com')
    assert result.startswith("Failed to navigate")
    assert "boom" in result

def test_get_page_content_text():
    result = web_explorer.get_page_content(mode="text")
    assert result == "Body Text"

def test_get_page_content_html_using_content_method():
    result = web_explorer.get_page_content(mode="html")
    # Should use page.content()
    assert result == "<html><body>Mock Content</body></html>"

def test_get_page_content_invalid_mode():
    result = web_explorer.get_page_content(mode="xml")
    assert result == "Invalid mode"

def test_click_element_css_selector():
    result = web_explorer.click_element("div#test")
    assert "Clicked: div#test" in result
    assert "New URL" in result

def test_click_element_text_selector():
    result = web_explorer.click_element("text=Click me")
    assert "Clicked: text=Click me" in result

def test_click_element_role_selector():
    result = web_explorer.click_element("role=button name=Submit")
    assert "Clicked: role=button name=Submit" in result

def test_click_element_error(monkeypatch):
    class BadPage(MockPage):
        def locator(self, selector):
            raise RuntimeError("bad locator")
    monkeypatch.setattr(web_explorer, 'get_page', lambda sid: BadPage())
    result = web_explorer.click_element("div")
    assert result.startswith("Failed to click")
    assert "bad locator" in result

def test_fill_input_success():
    result = web_explorer.fill_input("input#name", "John")
    assert result == "Filled 'input#name' with 'John'"

def test_fill_input_error(monkeypatch):
    class BadPage(MockPage):
        def locator(self, selector):
            raise RuntimeError("locator fail")
    monkeypatch.setattr(web_explorer, 'get_page', lambda sid: BadPage())
    result = web_explorer.fill_input("input#name", "John")
    assert result.startswith("Failed to fill input")
    assert "locator fail" in result

def test_screenshot_success():
    result = web_explorer.screenshot(full_page=True)
    expected_prefix = "data:image/png;base64,"
    assert result.startswith(expected_prefix)
    b64_part = result[len(expected_prefix):]
    assert base64.b64decode(b64_part) == b"pngbytes"

def test_end_browsing_page_success(monkeypatch):
    mock_close_page.called_with = None
    result = web_explorer.end_browsing_page(session_id="testsession")
    assert "Closed browser page" in result
    assert mock_close_page.called_with == "testsession"

def test_end_browsing_page_error(monkeypatch):
    def raise_err(sid):
        raise RuntimeError("close fail")
    monkeypatch.setattr(web_explorer, 'close_page', raise_err)
    result = web_explorer.end_browsing_page(session_id="s")
    assert result.startswith("Failed to close page")
    assert "close fail" in result
