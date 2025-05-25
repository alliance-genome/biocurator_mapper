"""End-to-end tests for ontology download functionality using Playwright."""
import pytest
import time
from playwright.sync_api import Page, expect
import os

# Get API key from environment or use test key
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "test-admin-key")
BASE_URL = os.getenv("STREAMLIT_URL", "http://localhost:8501")


class TestOntologyDownload:
    """Test suite for ontology download functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page):
        """Set up test environment before each test."""
        # Navigate to the main page
        page.goto(BASE_URL)
        
        # Wait for page to load
        page.wait_for_load_state("networkidle")
        
        # Enter API key - use the actual key from .env
        page.fill('input[type="password"]', "1234")
        
        # Press Enter to submit (Streamlit forms often work this way)
        page.press('input[type="password"]', "Enter")
        
        # Wait a bit for the unlock to process
        page.wait_for_timeout(1000)
        
        # Navigate to Ontology Update Management
        page.click('button:has-text("📥 Ontology Updates")')
        
        # Wait for page to load
        expect(page.locator('text=📥 Ontology Update Management')).to_be_visible()
        
        yield page
    
    def test_download_only_no_embedding(self, page: Page):
        """Test that ontology update only downloads file without parsing or embedding."""
        # Wait a bit for the page to fully load
        page.wait_for_timeout(1000)
        
        # The GO ontology section might already be expanded, let's look for the source URL input
        # Get the first text input which should be the GO source URL
        source_url_input = page.locator('input[type="text"]').first
        expect(source_url_input).to_have_value('http://purl.obolibrary.org/obo/go.json')
        
        # Click the first Update button (for GO)
        page.locator('button:has-text("🔄 Update from Source")').first.click()
        
        # Wait for update to start
        expect(page.locator('text=✅ Update started')).to_be_visible()
        
        # Wait for progress section to appear
        expect(page.locator('text=📥 Update Progress:')).to_be_visible()
        
        # Monitor progress
        # Should see download starting
        expect(page.locator('text=Starting download')).to_be_visible(timeout=10000)
        
        # Should see download progress
        expect(page.locator('text=/Downloaded.*MB/')).to_be_visible(timeout=30000)
        
        # IMPORTANT: Should NOT see any embedding-related messages
        # These should not appear at any point during the download
        # Be specific to avoid catching navigation buttons
        embedding_indicators = [
            'Generating embeddings',
            'Generating initial embeddings',
            'Creating embeddings',
            'Processing batch',
            'terms processed',
            'Embedding generation'
        ]
        
        for indicator in embedding_indicators:
            # Check that none of these appear within 5 seconds
            with pytest.raises(Exception):
                page.locator(f'text={indicator}').wait_for(timeout=5000)
        
        # Wait for download to complete
        expect(page.locator('text=Download completed!')).to_be_visible(timeout=60000)
        
        # Verify the logs don't contain embedding references
        logs_container = page.locator('[class*="logs"], [class*="recent"]').first()
        logs_text = logs_container.text_content()
        
        assert 'Download completed' in logs_text
        assert 'embedding' not in logs_text.lower()
        assert 'parsing' not in logs_text.lower()
        assert 'Processing batch' not in logs_text
    
    def test_download_progress_tracking(self, page: Page):
        """Test that download progress is properly tracked and displayed."""
        # Expand GO ontology
        page.click('text=🧬 GO - Gene Ontology')
        
        # Start download
        page.click('button:has-text("🔄 Update from Source")')
        
        # Verify progress tracking elements
        expect(page.locator('text=📥 Update Progress:')).to_be_visible()
        
        # Should show elapsed time
        expect(page.locator('text=/\\d+s elapsed/')).to_be_visible(timeout=5000)
        
        # Should show progress percentage
        expect(page.locator('[role="progressbar"], [class*="progress"]')).to_be_visible()
        
        # Should show download status in logs
        expect(page.locator('text=/Starting ontology update/')).to_be_visible(timeout=10000)
    
    def test_download_cancellation(self, page: Page):
        """Test that download can be cancelled."""
        # Expand GO ontology
        page.click('text=🧬 GO - Gene Ontology')
        
        # Start download
        page.click('button:has-text("🔄 Update from Source")')
        
        # Wait for progress to appear
        expect(page.locator('text=📥 Update Progress:')).to_be_visible()
        
        # Wait a bit for download to start
        time.sleep(2)
        
        # Click cancel button
        page.click('button:has-text("❌ Cancel Update")')
        
        # Verify cancellation
        expect(page.locator('text=/[Cc]ancelled/')).to_be_visible(timeout=10000)
    
    def test_downloaded_file_appears_in_embeddings_page(self, page: Page):
        """Test that downloaded files appear in the embeddings management page."""
        # Navigate to Embeddings Management
        page.click('button:has-text("🧠 Embeddings Management")')
        
        # Wait for page to load
        expect(page.locator('text=🧠 Embeddings Management')).to_be_visible()
        
        # GO ontology should be listed
        expect(page.locator('text=🧬 GO - Gene Ontology')).to_be_visible()
        
        # Expand GO section
        page.click('text=🧬 GO - Gene Ontology')
        
        # Should have option to generate embeddings
        expect(page.locator('button:has-text("🚀 Generate Embeddings")')).to_be_visible()
        
        # Should NOT be already generating embeddings
        with pytest.raises(Exception):
            page.locator('text=Generating initial embeddings').wait_for(timeout=2000)


def test_api_endpoint_returns_download_only(page: Page):
    """Test the API endpoint directly to ensure it only downloads."""
    import requests
    
    # Make direct API call
    response = requests.post(
        f"{BASE_URL.replace('8501', '8000')}/admin/update_ontology",
        headers={"X-API-Key": ADMIN_API_KEY},
        json={
            "ontology_name": "GO",
            "source_url": "http://purl.obolibrary.org/obo/go.json"
        }
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "update started"}
    
    # Check progress endpoint
    time.sleep(2)
    progress_response = requests.get(
        f"{BASE_URL.replace('8501', '8000')}/admin/update_progress/GO",
        headers={"X-API-Key": ADMIN_API_KEY}
    )
    
    progress_data = progress_response.json()
    
    # Should have download-related fields
    assert "download_percentage" in progress_data
    assert "download_bytes" in progress_data
    
    # Should NOT have embedding-related fields
    assert "embedding_status" not in progress_data
    assert "terms_processed" not in progress_data