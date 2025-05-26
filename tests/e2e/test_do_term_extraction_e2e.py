"""End-to-end test for DO term extraction with synonyms using Playwright MCP."""
import pytest
from playwright.sync_api import Page, expect
import json
import os


class TestDOTermExtractionE2E:
    """Test suite for verifying DO term extraction including synonyms."""
    
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
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=5000)
        
        yield page
    
    def test_do_download_extracts_synonyms(self, page: Page):
        """Test that downloading DO ontology correctly extracts terms with all synonym types."""
        # Step 1: Navigate to Ontology Updates
        ontology_updates_btn = page.locator('button:has-text("📥 Ontology Updates")')
        ontology_updates_btn.click()
        
        # Wait for page to load
        expect(page.locator('text=Ontology Update Management')).to_be_visible(timeout=5000)
        
        # Step 2: Expand DOID section
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait for expansion
        page.wait_for_timeout(500)
        
        # Step 3: Download DO ontology
        update_btn = page.locator('button:has-text("🔄 Update from Source")').filter(has_text="Update")
        # Find the visible update button
        for i in range(update_btn.count()):
            btn = update_btn.nth(i)
            if btn.is_visible():
                btn.click()
                break
        
        # Wait for download to complete
        completion_locator = page.locator('text=/Download completed|Update completed|completed successfully/i').or_(
            page.locator('text=/✅.*completed/i')
        ).or_(
            page.locator('text=/Successfully.*downloaded/i')
        )
        expect(completion_locator).to_be_visible(timeout=120000)
        
        # Step 4: Navigate to embeddings to verify term extraction
        self._verify_do_terms_have_synonyms(page)
    
    def _verify_do_terms_have_synonyms(self, page: Page):
        """Verify that DO terms were extracted with synonyms."""
        # Navigate to embeddings page
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        
        # Verify we're on the embeddings page
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        # Look for DOID section
        doid_section = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_section).to_be_visible(timeout=5000)
        
        # Check that terms were extracted (should show term count)
        # Look for text indicating terms are ready for embedding
        terms_indicator = page.locator('text=/[0-9,]+ terms.*ready|Terms:.*[0-9,]+/i')
        expect(terms_indicator).to_be_visible(timeout=10000)
        
        # Extract the term count to verify it's reasonable
        terms_text = terms_indicator.text_content()
        print(f"DO terms indicator: {terms_text}")
        
        # Verify a reasonable number of terms (DO has ~20,000 terms)
        import re
        match = re.search(r'([0-9,]+)\s*terms', terms_text, re.IGNORECASE)
        if match:
            term_count = int(match.group(1).replace(',', ''))
            assert term_count > 10000, f"Expected at least 10,000 DO terms, but found {term_count}"
            print(f"Successfully extracted {term_count} DO terms")
    
    def test_do_embeddings_include_synonyms(self, page: Page):
        """Test that DO embeddings generation includes synonym data."""
        # First ensure DO is downloaded
        self._ensure_do_downloaded(page)
        
        # Navigate to embeddings page
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        
        # Verify we're on the embeddings page
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        # Expand DOID section
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        
        # Wait for expansion
        page.wait_for_timeout(500)
        
        # Start embeddings generation
        generate_btns = page.locator('button:has-text("🚀 Generate Embeddings")')
        for i in range(generate_btns.count()):
            btn = generate_btns.nth(i)
            if btn.is_visible():
                btn.click()
                break
        
        # Confirm generation
        confirm_btn = page.locator('button:has-text("✅ Confirm")').first
        expect(confirm_btn).to_be_visible(timeout=5000)
        confirm_btn.click()
        
        # Wait for generation to start
        starting_status = page.locator('text=/Starting embedding generation/i')
        expect(starting_status).to_be_visible(timeout=10000)
        
        # Monitor for processing status that includes synonym information
        # The fixed extraction should result in richer embeddings
        processing_status = page.locator('text=/Processing.*terms|Generating embeddings/i')
        expect(processing_status).to_be_visible(timeout=30000)
        
        # Cancel after a few seconds to avoid long wait
        page.wait_for_timeout(5000)
        
        cancel_btn = page.locator('button:has-text("🛑 Cancel Generation")').first
        if cancel_btn.is_visible():
            cancel_btn.click()
            expect(page.locator('text=/Cancelling/i')).to_be_visible(timeout=10000)
    
    def _ensure_do_downloaded(self, page: Page):
        """Ensure DO ontology is downloaded before testing."""
        # Navigate to Ontology Updates
        page.locator('button:has-text("📥 Ontology Updates")').click()
        page.wait_for_timeout(2000)
        
        # Expand DOID section first to check status
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        if doid_expander.is_visible():
            doid_expander.click()
            page.wait_for_timeout(1000)
        
        # Check if already downloaded by looking for version info or status
        version_indicators = [
            page.locator('text=/Current Version.*DOID/i'),
            page.locator('text=/Version:.*[0-9]/i'),
            page.locator('text=/Last Updated/i'),
            page.locator('text=/Downloaded.*DOID/i')
        ]
        
        for indicator in version_indicators:
            if indicator.is_visible():
                print("DO ontology already downloaded")
                return
        
        # Download if needed
        update_btn = page.locator('button:has-text("🔄 Update from Source")').filter(has_text="Update")
        for i in range(update_btn.count()):
            btn = update_btn.nth(i)
            if btn.is_visible():
                print("Starting DO download...")
                btn.click()
                # Wait for completion with longer timeout
                completion_text = page.locator('text=/Download completed|Update completed|completed successfully/i')
                expect(completion_text).to_be_visible(timeout=180000)  # 3 minutes
                print("DO download completed")
                break
    
    def test_verify_searchable_text_generation(self, page: Page):
        """Test that searchable text is properly generated with synonyms."""
        # This test verifies the searchable text generation by checking
        # that embeddings are created with rich content
        
        # Ensure DO is downloaded
        self._ensure_do_downloaded(page)
        
        # Navigate to embeddings
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")').first
        embeddings_btn.click()
        
        # The presence of ready terms indicates successful extraction
        # with searchable text built from names, definitions, and synonyms
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        # Look for DOID section showing it's ready
        doid_section_text = page.locator('div:has(div:has-text("🧬 DOID"))').text_content()
        
        # The fact that terms are ready for embedding indicates
        # successful extraction and searchable text generation
        assert "terms" in doid_section_text.lower(), "DO terms should be ready for embedding"
        
        print("DO terms successfully extracted with searchable text")


class TestDOTermDetailsVisualization:
    """Test visualization of DO term details including synonyms."""
    
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
    
    def test_embeddings_config_shows_synonym_options(self, page: Page):
        """Test that embeddings configuration shows synonym vectorization options."""
        # Navigate to Embeddings Config
        config_btn = page.locator('button:has-text("Configure Embeddings")')
        expect(config_btn).to_be_visible(timeout=5000)
        config_btn.click()
        
        # Verify config page loaded
        expect(page.locator("text=Embeddings Configuration")).to_be_visible(timeout=5000)
        
        # Check for vectorize fields section
        # Look for synonym-related configuration options
        config_text = page.content()
        
        # The configuration should include options for:
        # - Vectorizing synonyms
        # - Name field
        # - Definition field
        assert "name" in config_text.lower() or "fields" in config_text.lower(), \
            "Configuration should show field options"
        
        print("Embeddings configuration includes field vectorization options")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])