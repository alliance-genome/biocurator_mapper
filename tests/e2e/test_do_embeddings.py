"""End-to-end test for DO (Disease Ontology) embeddings functionality."""
import pytest
from playwright.sync_api import Page, expect
import time


class TestDOEmbeddings:
    """Test suite for DO embeddings workflow."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str, admin_api_key: str):
        """Setup for each test - navigate and authenticate."""
        # Navigate to the app
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        # Enter API key
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        
        # Wait for authentication
        page.wait_for_timeout(2000)
        
        # Verify admin features are available
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=5000)
        
        yield page
    
    def test_complete_do_embeddings_generation(self, page: Page):
        """Test complete flow: download DO ontology, then generate embeddings."""
        # Step 1: Download DO ontology
        self._download_do_ontology(page)
        
        # Step 2: Navigate to embeddings page
        self._navigate_to_embeddings(page)
        
        # Step 3: Generate embeddings
        self._generate_embeddings(page)
        
        # Step 4: Verify completion
        self._verify_embeddings_completion(page)
    
    def _download_do_ontology(self, page: Page):
        """Download the DO ontology."""
        # Navigate to Ontology Updates
        ontology_updates_btn = page.locator('button:has-text("📥 Ontology Updates")')
        ontology_updates_btn.click()
        
        # Wait for page to load
        expect(page.locator('text=Ontology Update Management')).to_be_visible(timeout=5000)
        
        # Wait a moment for the page to fully load
        page.wait_for_timeout(1000)
        
        # Look for DOID expander and click it
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"').first
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait for expander to fully open
        page.wait_for_timeout(1000)
        
        # Find visible Update button - use nth to get the right one if multiple exist
        # Based on the screenshot, the DOID update button should be the second one
        all_update_buttons = page.locator('button:has-text("Update from Source")')
        
        # Find the visible button
        update_btn = None
        for i in range(all_update_buttons.count()):
            btn = all_update_buttons.nth(i)
            if btn.is_visible():
                update_btn = btn
                break
        
        if not update_btn:
            raise Exception("Could not find visible Update from Source button")
        
        # Click the button
        update_btn.click()
        
        # Wait a moment for the download to start
        page.wait_for_timeout(2000)
        
        # Take a screenshot to see what's happening
        page.screenshot(path="debug_download_progress.png")
        
        # Look for various completion or progress indicators
        # Check for progress messages first
        progress_indicators = [
            page.locator('text=/Downloading/i'),
            page.locator('text=/Processing/i'),
            page.locator('text=/Update in progress/i'),
            page.locator('text=/Starting download/i')
        ]
        
        # Wait for any progress indicator
        progress_found = False
        for indicator in progress_indicators:
            if indicator.is_visible():
                progress_found = True
                print(f"Progress indicator found: {indicator.text_content()}")
                break
        
        # Wait for download to complete - look for various success messages
        completion_selectors = [
            'text=/Download completed|Update completed|completed successfully/i',
            'text=/✅.*completed/i',
            'text=/Successfully.*downloaded/i',
            'text=/DOID.*updated/i',
            'text=/Saved.*DOID/i'
        ]
        
        completion_found = False
        for i in range(60):  # Check for 60 seconds
            for selector in completion_selectors:
                element = page.locator(selector)
                if element.is_visible():
                    completion_found = True
                    print(f"Completion found: {element.text_content()}")
                    break
            if completion_found:
                break
            page.wait_for_timeout(1000)
        
        # Take final screenshot
        page.screenshot(path="debug_download_final.png")
        
        if not completion_found:
            # Check for error messages
            error_element = page.locator('text=/Error|Failed|Exception/i').first
            if error_element.is_visible():
                raise Exception(f"Download failed with error: {error_element.text_content()}")
            else:
                raise Exception("Download did not complete within timeout")
    
    def _navigate_to_embeddings(self, page: Page):
        """Navigate to the embeddings management page."""
        # Click on Manage Embeddings button
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        
        # Verify we're on the embeddings page
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
    
    def _generate_embeddings(self, page: Page):
        """Generate embeddings for DO."""
        # Look for DOID expander using data-testid
        doid_expander_marker = page.locator('[data-testid="expander-DOID"]')
        if doid_expander_marker.count() > 0:
            # Click the expander to open it
            doid_embedding_section = page.locator('text=🧬 DOID - Disease Ontology').first
            if doid_embedding_section.is_visible():
                doid_embedding_section.click()
        
        # Find Generate Embeddings button using data-testid marker
        generate_btn_marker = page.locator('[data-testid="button-generate-embeddings-DOID"]')
        expect(generate_btn_marker).to_be_attached(timeout=5000)
        
        # Click the actual button (next sibling of the marker)
        generate_btn = page.locator('button:has-text("🚀 Generate Embeddings")', has=page.locator('[data-testid="button-generate-embeddings-DOID"]'))
        if not generate_btn.is_visible():
            # Fallback to finding by key
            generate_btn = page.locator('button[id*="gen_embed_btn_DOID"]')
        
        expect(generate_btn).to_be_visible(timeout=5000)
        generate_btn.click()
        
        # Wait for confirmation dialog and confirm
        confirm_btn = page.locator('button:has-text("✅ Confirm")').first
        expect(confirm_btn).to_be_visible(timeout=5000)
        confirm_btn.click()
    
    def _verify_embeddings_completion(self, page: Page):
        """Verify embeddings generation completes successfully."""
        # Check for embedding progress marker
        progress_marker = page.locator('[data-testid="embedding-progress-DOID"]')
        expect(progress_marker).to_be_attached(timeout=10000)
        
        # Wait for initial status
        status_marker = page.locator('[data-testid="text-status-DOID"]')
        expect(status_marker).to_be_attached(timeout=5000)
        
        # Monitor status progression
        expect(page.locator('[data-testid="text-status-DOID"][data-status="starting"]')).to_be_attached(timeout=10000)
        
        # Wait for processing status
        expect(page.locator('[data-testid="text-status-DOID"][data-status*="processing"]').or_(
            page.locator('[data-testid="text-status-DOID"][data-status*="embedding"]')
        )).to_be_attached(timeout=30000)
        
        # Monitor progress bar
        progress_bar = page.locator('[data-testid="progressbar-DOID"]')
        if progress_bar.count() > 0:
            # Wait for progress to reach at least 50%
            expect(progress_bar).to_have_attribute("data-progress", "50", timeout=60000)
        
        # Wait for completion status
        expect(page.locator('[data-testid="text-status-DOID"][data-status="completed"]')).to_be_attached(timeout=300000)
        
        # Also check for visible completion message
        expect(page.locator('text=/Embedding generation completed|Embeddings generated successfully/i')).to_be_visible(timeout=5000)
        
        # Verify no error status
        error_status = page.locator('[data-testid="text-status-DOID"][data-status="failed"]')
        expect(error_status).not_to_be_attached()


class TestDOEmbeddingsCancellation:
    """Test cancellation of embeddings generation."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str, admin_api_key: str):
        """Setup for each test."""
        # Navigate and authenticate
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        page.wait_for_timeout(2000)
        
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=5000)
        
        # Ensure DO is downloaded first
        self._ensure_do_downloaded(page)
        
        yield page
    
    def _ensure_do_downloaded(self, page: Page):
        """Ensure DO ontology is downloaded before testing embeddings."""
        # Check if we need to download
        page.locator('button:has-text("📥 Ontology Updates")').click()
        page.wait_for_timeout(2000)
        
        # Look for existing version info
        if page.locator('text=/Current Version.*DOID/i').is_visible():
            # Already downloaded
            return
        
        # Download if needed
        doid_section = page.locator('div:has(div:has-text("🧬 DOID - Disease Ontology"))')
        update_btn = doid_section.locator('button:has-text("🔄 Update from Source")').first
        
        if update_btn.is_visible():
            update_btn.click()
            completion_text = page.locator('text=/Download completed|Update completed|completed successfully/i')
            expect(completion_text).to_be_visible(timeout=60000)
            page.wait_for_timeout(2000)
    
    def test_cancel_embeddings_generation(self, page: Page):
        """Test cancelling embeddings generation."""
        # Navigate to embeddings page
        page.locator('button:has-text("🧠 Manage Embeddings")').first.click()
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        # Start embeddings generation using data-testid
        generate_btn_marker = page.locator('[data-testid="button-generate-embeddings-DOID"]')
        if generate_btn_marker.count() > 0:
            # Open expander first if needed
            doid_expander = page.locator('text=🧬 DOID - Disease Ontology').first
            if doid_expander.is_visible():
                doid_expander.click()
        
        # Click generate button
        generate_btn = page.locator('button[id*="gen_embed_btn_DOID"]')
        expect(generate_btn).to_be_visible(timeout=5000)
        generate_btn.click()
        
        # Confirm
        confirm_btn = page.locator('button:has-text("✅ Confirm")').first
        expect(confirm_btn).to_be_visible(timeout=5000)
        confirm_btn.click()
        
        # Wait for generation to start - check status marker
        expect(page.locator('[data-testid="text-status-DOID"][data-status="starting"]')).to_be_attached(timeout=10000)
        
        # Wait for processing to begin
        expect(page.locator('[data-testid="text-status-DOID"][data-status*="processing"]').or_(
            page.locator('[data-testid="text-status-DOID"][data-status*="embedding"]')
        )).to_be_attached(timeout=20000)
        
        # Find and click cancel button using data-testid
        cancel_btn_marker = page.locator('[data-testid="button-cancel-embeddings-DOID"]')
        expect(cancel_btn_marker).to_be_attached(timeout=5000)
        
        cancel_btn = page.locator('button[id*="cancel_embedding_DOID"]')
        expect(cancel_btn).to_be_visible(timeout=5000)
        cancel_btn.click()
        
        # Verify cancellation status
        expect(page.locator('[data-testid="text-status-DOID"][data-status="cancelling"]')).to_be_attached(timeout=10000)
        
        # Eventually should show cancelled status
        expect(page.locator('[data-testid="text-status-DOID"][data-status="cancelled"]')).to_be_attached(timeout=30000)
        
        # Verify no completion status
        completion_status = page.locator('[data-testid="text-status-DOID"][data-status="completed"]')
        expect(completion_status).not_to_be_attached()


class TestEmbeddingsConfiguration:
    """Test embeddings configuration interaction."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str, admin_api_key: str):
        """Setup for each test."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        page.wait_for_timeout(2000)
        
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=5000)
        
        yield page
    
    def test_modify_embeddings_configuration(self, page: Page):
        """Test modifying embeddings configuration."""
        # Navigate to Embeddings Config
        config_btn = page.locator('button:has-text("Configure Embeddings")')
        expect(config_btn).to_be_visible(timeout=5000)
        config_btn.click()
        
        # Verify config page loaded
        expect(page.locator("text=Embeddings Configuration")).to_be_visible(timeout=5000)
        
        # Look for model selection using data-testid
        model_select_marker = page.locator('[data-testid="select-embedding-model"]')
        expect(model_select_marker).to_be_attached(timeout=5000)
        
        # Find the actual select element
        model_select = page.locator('div[role="combobox"]').filter(has_text="ada").or_(
            page.locator('select').first
        )
        
        if model_select.is_visible():
            # Click to open dropdown
            model_select.click()
            
            # Select a different model
            new_model_option = page.locator('div[role="option"]:has-text("text-embedding-3-small")').or_(
                page.locator('option:has-text("text-embedding-3-small")')
            )
            if new_model_option.is_visible():
                new_model_option.click()
        
        # Look for batch size input using data-testid
        batch_size_marker = page.locator('[data-testid="input-batch-size"]')
        if batch_size_marker.count() > 0:
            # Find the actual input
            batch_input = page.locator('input[type="number"][value*="100"]').or_(
                page.locator('input[type="number"]').filter(has_text="Batch Size")
            ).first
            if batch_input.is_visible():
                batch_input.fill("50")
        
        # Look for save button using data-testid
        save_btn_marker = page.locator('[data-testid="button-save-embedding-config"]')
        expect(save_btn_marker).to_be_attached(timeout=5000)
        
        # Click the actual save button
        save_btn = page.locator('button:has-text("💾 Save Configuration")')
        expect(save_btn).to_be_visible(timeout=5000)
        save_btn.click()
        
        # Verify save success
        expect(page.locator('text=/Configuration saved|Successfully saved|✅.*saved/i')).to_be_visible(timeout=5000)
        
        # Test configuration button
        test_btn_marker = page.locator('[data-testid="button-test-embedding-config"]')
        if test_btn_marker.count() > 0:
            test_btn = page.locator('button:has-text("🔄 Test Configuration")')
            if test_btn.is_visible():
                test_btn.click()
                # Wait for test results
                expect(page.locator('text=/Test.*successful|Config.*OK|✅.*Config/i')).to_be_visible(timeout=10000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])