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
        
        # Wait for authentication by checking for admin panel visibility
        
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
        
        # Page load is confirmed by the previous expect
        
        # Look for DOID expander header and click it
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait a moment for expansion
        page.wait_for_timeout(500)
        
        # Find Update button - it should be visible after expanding
        update_btn = page.locator('button:has-text("🔄 Update from Source")').filter(has_text="Update")
        # Find the one that's visible
        for i in range(update_btn.count()):
            btn = update_btn.nth(i)
            if btn.is_visible():
                btn.click()
                break
        
        # Wait for download to start by checking for progress indicators
        
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
        
        # Wait for any completion indicator with proper timeout
        completion_locator = page.locator('text=/Download completed|Update completed|completed successfully/i').or_(
            page.locator('text=/✅.*completed/i')
        ).or_(
            page.locator('text=/Successfully.*downloaded/i')
        ).or_(
            page.locator('text=/DOID.*updated/i')
        ).or_(
            page.locator('text=/Saved.*DOID/i')
        )
        
        expect(completion_locator).to_be_visible(timeout=60000)
        
        # Take final screenshot
        page.screenshot(path="debug_download_final.png")
        
        # No need for additional error checking as expect will fail with timeout if not found
    
    def _navigate_to_embeddings(self, page: Page):
        """Navigate to the embeddings management page."""
        # Click on Manage Embeddings button
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        
        # Verify we're on the embeddings page
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
    
    def _generate_embeddings(self, page: Page):
        """Generate embeddings for DO."""
        # Look for DOID expander and click it
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait a moment for expansion
        page.wait_for_timeout(500)
        
        # Find Generate Embeddings button - look for visible one
        generate_btns = page.locator('button:has-text("🚀 Generate Embeddings")')
        for i in range(generate_btns.count()):
            btn = generate_btns.nth(i)
            if btn.is_visible():
                btn.click()
                break
        
        # Wait for confirmation dialog and confirm
        confirm_btn = page.locator('button:has-text("✅ Confirm")').first
        expect(confirm_btn).to_be_visible(timeout=5000)
        confirm_btn.click()
    
    def _verify_embeddings_completion(self, page: Page):
        """Verify embeddings generation completes successfully."""
        # Check for status messages directly (containers don't have data-testid)
        # Wait for starting status
        starting_status = page.locator('text=/Starting embedding generation/i')
        expect(starting_status).to_be_visible(timeout=10000)
        
        # Monitor status messages
        # Wait for processing/embedding status
        processing_status = page.locator('text=/Processing terms|Generating embeddings/i')
        expect(processing_status).to_be_visible(timeout=30000)
        
        # Wait for completion (this could take a while depending on ontology size)
        completion_status = page.locator('text=/Embedding generation completed|Successfully created/i')
        expect(completion_status).to_be_visible(timeout=300000)
        
        # Verify no errors
        error_status = page.locator('text=/Failed|Error/i')
        expect(error_status).not_to_be_visible()


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
    
    def test_cancel_embeddings_generation(self, page: Page):
        """Test cancelling embeddings generation."""
        # Navigate to embeddings page
        page.locator('button:has-text("🧠 Manage Embeddings")').first.click()
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        # Look for DOID expander and click it
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait a moment for expansion
        page.wait_for_timeout(500)
        
        # Find and click generate button
        generate_btns = page.locator('button:has-text("🚀 Generate Embeddings")')
        for i in range(generate_btns.count()):
            btn = generate_btns.nth(i)
            if btn.is_visible():
                btn.click()
                break
        
        # Confirm
        confirm_btn = page.locator('button:has-text("✅ Confirm")').first
        expect(confirm_btn).to_be_visible(timeout=5000)
        confirm_btn.click()
        
        # Wait for generation to start
        starting_status = page.locator('text=/Starting embedding generation/i')
        expect(starting_status).to_be_visible(timeout=10000)
        
        # Wait for processing to begin
        processing_status = page.locator('text=/Processing terms|Generating embeddings/i')
        expect(processing_status).to_be_visible(timeout=20000)
        
        # Find and click cancel button
        cancel_btn = page.locator('button:has-text("🛑 Cancel Generation")').first
        expect(cancel_btn).to_be_visible(timeout=5000)
        cancel_btn.click()
        
        # Verify cancellation status
        expect(page.locator('text=/Cancelling/i')).to_be_visible(timeout=10000)
        
        # Eventually should show cancelled status
        expect(page.locator('text=/Embedding generation cancelled/i')).to_be_visible(timeout=30000)
        
        # Verify no completion status
        completion_status = page.locator('text=/Embedding generation completed/i')
        expect(completion_status).not_to_be_visible()


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
        
        # Find model selection using data-testid
        model_select = page.locator('[data-testid="select-embedding-model"]')
        expect(model_select).to_be_visible(timeout=5000)
        
        # Click to open dropdown and select a different model
        model_select.click()
        # Select text-embedding-3-small option
        new_model_option = page.locator('div[role="option"]:has-text("text-embedding-3-small")')
        expect(new_model_option).to_be_visible(timeout=5000)
        new_model_option.click()
        
        # Find batch size input using data-testid
        batch_input = page.locator('[data-testid="input-batch-size"]')
        expect(batch_input).to_be_visible(timeout=5000)
        batch_input.fill("50")
        
        # Find save button using data-testid
        save_btn = page.locator('[data-testid="button-save-embedding-config"]')
        expect(save_btn).to_be_visible(timeout=5000)
        save_btn.click()
        
        # Verify save success
        expect(page.locator('text=/Configuration saved|Successfully saved|✅.*saved/i')).to_be_visible(timeout=5000)
        
        # Test configuration button
        test_btn = page.locator('[data-testid="button-test-embedding-config"]')
        expect(test_btn).to_be_visible(timeout=5000)
        test_btn.click()
        
        # Wait for test results
        expect(page.locator('text=/Test.*successful|Config.*valid|✅/i')).to_be_visible(timeout=10000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])