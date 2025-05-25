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
        page.wait_for_timeout(2000)
        
        # Look for DOID section - may need to expand
        doid_header = page.locator('text=🧬 DOID - Disease Ontology').first
        if doid_header.is_visible():
            doid_header.click()
            page.wait_for_timeout(1000)
        
        # Find and click the Update button for DOID
        # Look for the update button within the DOID section
        doid_section = page.locator('div:has(div:has-text("🧬 DOID - Disease Ontology"))')
        update_btn = doid_section.locator('button:has-text("🔄 Update from Source")').first
        
        # Wait for button to be visible and click
        expect(update_btn).to_be_visible(timeout=5000)
        update_btn.click()
        
        # Wait for download to complete
        # Look for completion indicators
        completion_text = page.locator('text=/Download completed|Update completed|completed successfully|✅/i')
        expect(completion_text).to_be_visible(timeout=60000)  # 60s timeout for download
        
        # Extra wait for UI to stabilize
        page.wait_for_timeout(2000)
    
    def _navigate_to_embeddings(self, page: Page):
        """Navigate to the embeddings management page."""
        # Click on Manage Embeddings button
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        page.wait_for_timeout(2000)
        
        # Verify we're on the embeddings page
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
    
    def _generate_embeddings(self, page: Page):
        """Generate embeddings for DO."""
        # Find DOID section on embeddings page - may need to expand
        doid_embedding_section = page.locator('text=🧬 DOID - Disease Ontology').first
        if doid_embedding_section.is_visible():
            doid_embedding_section.click()
            page.wait_for_timeout(1000)
        
        # Find Generate Embeddings button within DOID section
        doid_section = page.locator('div:has-text("Ontology: DOID")')
        generate_btn = doid_section.locator('button:has-text("🚀 Generate Embeddings")').first
        
        # Click the button
        expect(generate_btn).to_be_visible(timeout=5000)
        generate_btn.click()
        
        # Wait for confirmation dialog and confirm if present
        try:
            confirm_btn = page.locator('button:has-text("Yes, Generate")').or_(page.locator('button:has-text("Confirm")')).first
            if confirm_btn.is_visible(timeout=2000):
                confirm_btn.click()
        except:
            pass  # No confirmation dialog
    
    def _verify_embeddings_completion(self, page: Page):
        """Verify embeddings generation completes successfully."""
        # Wait for progress indicators
        expect(page.locator('text=/Starting embedding generation|Loading ontology data|Parsing ontology terms/i')).to_be_visible(timeout=10000)
        
        # Monitor for completion (up to 5 minutes for DO)
        completion_text = page.locator('text=/Embedding generation completed|Embeddings generated successfully|✅.*completed/i')
        expect(completion_text).to_be_visible(timeout=300000)  # 5 minute timeout
        
        # Verify no errors
        error_text = page.locator('text=/Embedding generation failed|Error generating embeddings|Failed to generate/i')
        expect(error_text).not_to_be_visible()
        
        # Look for statistics if available
        stats_text = page.locator('text=/Generated embeddings for.*terms|processed.*terms/i')
        if stats_text.is_visible():
            print(f"Embedding stats: {stats_text.text_content()}")


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
        page.wait_for_timeout(2000)
        
        # Start embeddings generation
        doid_section = page.locator('div:has-text("Ontology: DOID")')
        generate_btn = doid_section.locator('button:has-text("🚀 Generate Embeddings")').first
        generate_btn.click()
        
        # Confirm if needed
        try:
            confirm_btn = page.locator('button:has-text("Yes, Generate")').first
            if confirm_btn.is_visible(timeout=2000):
                confirm_btn.click()
        except:
            pass
        
        # Wait for generation to start
        expect(page.locator('text=/Starting embedding generation|Processing/i')).to_be_visible(timeout=10000)
        
        # Click cancel button
        cancel_btn = page.locator('button:has-text("❌ Cancel Embeddings")').first
        expect(cancel_btn).to_be_visible(timeout=5000)
        cancel_btn.click()
        
        # Verify cancellation
        expect(page.locator('text=/Cancellation requested|cancelled|cancelling/i')).to_be_visible(timeout=10000)
        
        # Verify generation stops (no completion message)
        page.wait_for_timeout(5000)
        completion_text = page.locator('text=/Embeddings generated successfully/i')
        expect(completion_text).not_to_be_visible()


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
        page.locator('text=⚙️ Embeddings Config').click()
        page.wait_for_timeout(2000)
        
        # Verify config page loaded
        expect(page.locator("text=Embeddings Configuration")).to_be_visible()
        
        # Find model selection
        model_select = page.locator('select').filter(has_text="text-ada-002").or_(
            page.locator('div[data-baseweb="select"]').first
        )
        
        # Change model if possible
        if model_select.is_visible():
            # Get current value
            current_model = model_select.input_value() if hasattr(model_select, 'input_value') else "text-ada-002"
            
            # Try to change to a different model
            new_model = "text-embedding-3-small" if current_model != "text-embedding-3-small" else "text-ada-002"
            
            # For Streamlit selects, we may need to click and select
            model_select.click()
            page.locator(f'text={new_model}').click()
            page.wait_for_timeout(1000)
        
        # Modify batch size
        batch_input = page.locator('input[type="number"]').filter(has=page.locator('text=/batch.*size/i'))
        if batch_input.is_visible():
            batch_input.fill("50")
        
        # Save configuration
        save_btn = page.locator('button:has-text("💾 Save Configuration")')
        save_btn.click()
        
        # Verify save success
        expect(page.locator('text=/Configuration saved|Successfully saved/i')).to_be_visible(timeout=5000)
        
        # Test Reset to Defaults
        reset_btn = page.locator('button:has-text("🔄 Reset to Defaults")')
        if reset_btn.is_visible():
            reset_btn.click()
            page.wait_for_timeout(1000)
            
            # Verify reset message
            expect(page.locator('text=/Reset to defaults|Configuration reset/i')).to_be_visible(timeout=5000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])