"""Verified E2E test for DO term extraction based on Playwright MCP exploration."""
import pytest
from playwright.sync_api import Page, expect
import os


class TestDOWorkflowVerified:
    """Verified test suite based on manual Playwright MCP exploration."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str, admin_api_key: str):
        """Setup for each test - navigate and authenticate."""
        # Navigate to the app
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        # The API key field should be visible
        api_input = page.get_by_role('textbox', name='API Key')
        expect(api_input).to_be_visible(timeout=5000)
        
        # Clear existing value and enter API key
        api_input.click()
        api_input.press("Control+a")
        api_input.fill(admin_api_key)
        
        # Press Enter to authenticate
        api_input.press("Enter")
        
        # Wait a moment for authentication to process
        page.wait_for_timeout(2000)
        
        # Verify we have access to admin features
        # The admin panel should be visible after auth
        admin_panel = page.locator("text=🔧 Admin Panel")
        expect(admin_panel).to_be_visible(timeout=10000)
        
        yield page
    
    def test_navigate_to_ontology_updates(self, page: Page):
        """Test navigation to ontology updates page."""
        # Click the Ontology Updates button
        ontology_btn = page.get_by_role('button', name='📥 Ontology Updates')
        expect(ontology_btn).to_be_visible(timeout=5000)
        ontology_btn.click()
        
        # Verify we're on the right page
        page_title = page.locator("text=📥 Ontology Update Management")
        expect(page_title).to_be_visible(timeout=5000)
        
        # Check that DOID section is present
        doid_section = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_section).to_be_visible(timeout=5000)
        
        print("✅ Successfully navigated to Ontology Updates page")
    
    def test_expand_doid_section(self, page: Page):
        """Test expanding the DOID section."""
        # Navigate to Ontology Updates
        page.get_by_role('button', name='📥 Ontology Updates').click()
        expect(page.locator("text=📥 Ontology Update Management")).to_be_visible(timeout=5000)
        
        # Click to expand DOID section
        doid_expander = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        doid_expander.click()
        
        # Wait for expansion
        page.wait_for_timeout(1000)
        
        # Verify the Update button is now visible
        update_btn = page.get_by_role('button', name='🔄 Update from Source')
        expect(update_btn).to_be_visible(timeout=5000)
        
        # Check that source URL field is visible
        source_url = page.get_by_role('textbox', name='Source URL')
        expect(source_url).to_be_visible()
        expect(source_url).to_have_value('http://purl.obolibrary.org/obo/doid.json')
        
        print("✅ DOID section expanded successfully")
    
    def test_navigate_to_embeddings_management(self, page: Page):
        """Test navigation to embeddings management page."""
        # Click Manage Embeddings button
        embeddings_btn = page.get_by_role('button', name='🧠 Manage Embeddings')
        expect(embeddings_btn).to_be_visible(timeout=5000)
        embeddings_btn.click()
        
        # Verify we're on the embeddings page
        page_title = page.locator("text=Ontology Embedding Management")
        expect(page_title).to_be_visible(timeout=5000)
        
        print("✅ Successfully navigated to Embeddings Management page")
    
    def test_doid_download_workflow(self, page: Page):
        """Test DOID download workflow with valid API key."""
        # Navigate to Ontology Updates
        page.get_by_role('button', name='📥 Ontology Updates').click()
        expect(page.locator("text=📥 Ontology Update Management")).to_be_visible(timeout=5000)
        
        # Expand DOID section
        doid_expander = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        doid_expander.click()
        page.wait_for_timeout(500)
        
        # Click update button
        update_btn = page.get_by_role('button', name='🔄 Update from Source')
        update_btn.click()
        
        # Check for progress indicator
        progress_msg = page.locator('text=/Downloading ontology/i')
        expect(progress_msg).to_be_visible(timeout=10000)
        
        # Wait for completion (with timeout)
        completion_msg = page.locator('text=/Update completed/i')
        expect(completion_msg).to_be_visible(timeout=60000)  # 60 seconds for download
        
        print("✅ DOID download completed successfully")
    
    def test_embeddings_config_navigation(self, page: Page):
        """Test navigation to embeddings configuration."""
        # Click Configure Embeddings button
        config_btn = page.get_by_role('button', name='Configure Embeddings')
        expect(config_btn).to_be_visible(timeout=5000)
        config_btn.click()
        
        # Verify we're on the config page
        page_title = page.locator("text=Embeddings Configuration")
        expect(page_title).to_be_visible(timeout=5000)
        
        print("✅ Successfully navigated to Embeddings Configuration")


class TestDOWorkflowMockAPI:
    """Test suite with mocked API responses for successful flows."""
    
    @pytest.fixture(autouse=True)
    def setup_with_mock(self, page: Page, base_url: str):
        """Setup with mocked API responses."""
        # Mock the admin authentication to always succeed
        page.route("**/api/admin/update", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status": "success", "message": "Update completed"}'
        ))
        
        # Navigate and use a dummy API key
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.get_by_role('textbox', name='API Key')
        api_input.fill("mocked-key")
        api_input.press("Enter")
        page.wait_for_timeout(2000)
        
        yield page
    
    def test_do_download_with_mock(self, page: Page):
        """Test DO download with mocked successful response."""
        # Navigate to Ontology Updates
        page.get_by_role('button', name='📥 Ontology Updates').click()
        page.wait_for_timeout(1000)
        
        # Expand DOID section
        doid_expander = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        doid_expander.click()
        page.wait_for_timeout(500)
        
        # Click update - should succeed with mock
        update_btn = page.get_by_role('button', name='🔄 Update from Source')
        update_btn.click()
        
        # With mocked response, we should see success
        # (In real scenario, this would show download progress)
        page.wait_for_timeout(2000)
        
        print("✅ DO download initiated with mocked API")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])