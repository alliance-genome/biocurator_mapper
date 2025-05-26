"""Pytest configuration for Playwright E2E tests."""
import pytest
from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@pytest.fixture(scope="session")
def browser():
    """Create a browser instance for the test session."""
    with sync_playwright() as p:
        # Use headless mode by default, unless HEADED env var is set
        headless = os.getenv("HEADED", "false").lower() == "false"
        browser = p.chromium.launch(headless=headless)
        yield browser
        browser.close()


@pytest.fixture(scope="function")
def page(browser):
    """Create a new page for each test."""
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture(scope="session")
def base_url():
    """Get the base URL for the application."""
    return os.getenv("STREAMLIT_URL", "http://localhost:8501")


@pytest.fixture(scope="session")
def api_url():
    """Get the API URL for the FastAPI backend."""
    return os.getenv("FASTAPI_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def admin_api_key():
    """Get the admin API key from environment variables.
    
    First tries to get from ADMIN_API_KEY env var (which includes .env file).
    Falls back to a placeholder if not found (tests will fail with invalid key).
    """
    api_key = os.getenv("ADMIN_API_KEY")
    if not api_key:
        pytest.skip("ADMIN_API_KEY not found in environment variables or .env file")
    return api_key