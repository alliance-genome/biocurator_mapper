"""Complete DO workflow E2E tests based on manual Playwright MCP verification.

This test suite covers the complete DO workflow from download to embeddings generation,
based on actual manual testing using Playwright MCP.
"""
import pytest
from playwright.sync_api import Page, expect
import time


class TestCompleteDOWorkflow:
    """Complete DO workflow tests based on manual MCP verification."""
    
    @pytest.fixture(autouse=True)
    def setup(self, page: Page, base_url: str, admin_api_key: str):
        """Setup for each test - navigate and authenticate."""
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        # Enter API key for authentication
        api_input = page.get_by_role('textbox', name='API Key')
        expect(api_input).to_be_visible(timeout=5000)
        api_input.fill(admin_api_key)
        
        # Wait for authentication to process
        page.wait_for_timeout(2000)
        
        # Verify admin panel is accessible
        admin_panel = page.locator("text=🔧 Admin Panel")
        expect(admin_panel).to_be_visible(timeout=10000)
        
        yield page
    
    def test_complete_do_download_workflow(self, page: Page):
        """Test complete DO download workflow from start to finish."""
        # Step 1: Navigate to Ontology Updates
        ontology_btn = page.get_by_role('button', name='📥 Ontology Updates')
        ontology_btn.click()
        
        # Verify we're on the ontology management page
        page_title = page.locator("text=📥 Ontology Update Management")
        expect(page_title).to_be_visible(timeout=5000)
        
        # Step 2: Expand DOID section
        doid_expander = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_expander).to_be_visible(timeout=5000)
        doid_expander.click()
        page.wait_for_timeout(1000)
        
        # Verify DOID section expanded with details
        source_url = page.get_by_role('textbox', name='Source URL')
        expect(source_url).to_be_visible()
        expect(source_url).to_have_value('http://purl.obolibrary.org/obo/doid.json')
        
        update_btn = page.get_by_role('button', name='🔄 Update from Source')
        expect(update_btn).to_be_visible()
        
        # Step 3: Start download
        update_btn.click()
        
        # Step 4: Verify download progress appears
        progress_section = page.locator("text=📥 Update Progress:")
        expect(progress_section).to_be_visible(timeout=10000)
        
        # Check for download progress indicator
        progress_msg = page.locator('text=/Downloading ontology/i')
        expect(progress_msg).to_be_visible(timeout=10000)
        
        # Step 5: Wait for download completion
        # Based on manual testing: download takes ~27 seconds
        completion_msg = page.locator('text=/Update completed/i')
        expect(completion_msg).to_be_visible(timeout=60000)  # 60 second timeout
        
        # Step 6: Verify download success details
        progress_bar = page.locator('progressbar')
        expect(progress_bar).to_have_attribute('aria-valuenow', '100', timeout=5000)
        
        # Check for success logs
        logs_section = page.locator('text=/Recent Logs/i')
        expect(logs_section).to_be_visible()
        
        print("✅ Complete DO download workflow verified")
    
    def test_embeddings_management_workflow(self, page: Page):
        """Test embeddings management workflow after download."""
        # Navigate to Embeddings Management
        embeddings_btn = page.get_by_role('button', name='🧠 Manage Embeddings')
        embeddings_btn.click()
        
        # Verify we're on embeddings page
        page_title = page.locator("text=🧠 Ontology Embedding Management")
        expect(page_title).to_be_visible(timeout=5000)
        
        # Check configuration is displayed
        config_alert = page.locator('text=/Current Configuration/i')
        expect(config_alert).to_be_visible()
        
        # Verify model configuration
        model_text = page.locator('text=text-ada-002')
        expect(model_text).to_be_visible()
        
        batch_size_text = page.locator('text=100')
        expect(batch_size_text).to_be_visible()
        
        fields_text = page.locator('text=name, definition, synonyms')
        expect(fields_text).to_be_visible()
        
        print("✅ Embeddings management interface verified")
    
    def test_verify_downloads_functionality(self, page: Page):
        """Test the Verify Downloads functionality that refreshes availability."""
        # Navigate to Embeddings Management
        page.get_by_role('button', name='🧠 Manage Embeddings').click()
        
        # Click Verify Downloads button
        verify_btn = page.get_by_role('button', name='🔍 Verify Downloads')
        expect(verify_btn).to_be_visible()
        verify_btn.click()
        
        # Wait for verification to complete
        # Based on manual testing, this resolves connection issues
        page.wait_for_timeout(10000)
        
        # After verification, DOID should be available
        available_section = page.locator('text=/Available Ontologies for Embedding/i')
        expect(available_section).to_be_visible(timeout=15000)
        
        # DOID should appear as expandable section
        doid_embedding_section = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_embedding_section).to_be_visible(timeout=10000)
        
        print("✅ Verify Downloads functionality verified")
    
    def test_doid_embeddings_generation_interface(self, page: Page):
        """Test DOID embeddings generation interface and confirmation dialog."""
        # Navigate to Embeddings Management
        page.get_by_role('button', name='🧠 Manage Embeddings').click()
        
        # Verify downloads to make DOID available
        verify_btn = page.get_by_role('button', name='🔍 Verify Downloads')
        verify_btn.click()
        page.wait_for_timeout(10000)
        
        # Expand DOID embeddings section
        doid_embedding_section = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_embedding_section).to_be_visible(timeout=15000)
        doid_embedding_section.click()
        page.wait_for_timeout(1000)
        
        # Verify embeddings details are shown
        ontology_text = page.locator('text=Ontology: DOID')
        expect(ontology_text).to_be_visible()
        
        last_updated = page.locator('text=/Last Updated:/i')
        expect(last_updated).to_be_visible()
        
        embedding_model = page.locator('text=Embedding Model: text-ada-002')
        expect(embedding_model).to_be_visible()
        
        estimated_cost = page.locator('text=/Estimated Cost:/i')
        expect(estimated_cost).to_be_visible()
        
        # Find and click Generate Embeddings button
        generate_btn = page.get_by_role('button', name='🚀 Generate Embeddings')
        expect(generate_btn).to_be_visible()
        generate_btn.click()
        
        # Verify confirmation dialog appears
        cost_warning = page.locator('text=/This will incur costs/i')
        expect(cost_warning).to_be_visible(timeout=5000)
        
        confirm_question = page.locator('text=/Generate embeddings for DOID/i')
        expect(confirm_question).to_be_visible()
        
        cost_estimate = page.locator('text=/Est. cost:/i')
        expect(cost_estimate).to_be_visible()
        
        # Verify confirm and cancel buttons
        confirm_btn = page.get_by_role('button', name='✅ Confirm')
        expect(confirm_btn).to_be_visible()
        
        cancel_btn = page.get_by_role('button', name='❌ Cancel')
        expect(cancel_btn).to_be_visible()
        
        print("✅ DOID embeddings generation interface verified")
    
    def test_embeddings_generation_progress_interface(self, page: Page):
        """Test embeddings generation progress interface (without running full generation)."""
        # Navigate to Embeddings Management and start generation
        page.get_by_role('button', name='🧠 Manage Embeddings').click()
        
        # Verify downloads and expand DOID section
        verify_btn = page.get_by_role('button', name='🔍 Verify Downloads')
        verify_btn.click()
        page.wait_for_timeout(10000)
        
        doid_embedding_section = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_embedding_section).to_be_visible(timeout=15000)
        doid_embedding_section.click()
        page.wait_for_timeout(1000)
        
        # Start generation
        generate_btn = page.get_by_role('button', name='🚀 Generate Embeddings')
        generate_btn.click()
        
        # Confirm to start process
        confirm_btn = page.get_by_role('button', name='✅ Confirm')
        confirm_btn.click()
        
        # Wait for progress interface to appear
        page.wait_for_timeout(15000)
        
        # Verify progress section appears
        progress_section = page.locator('text=🤖 Embedding Progress')
        expect(progress_section).to_be_visible(timeout=20000)
        
        # Check for initialization status
        # Based on manual testing: shows "🔧 Initializing"
        initializing_status = page.locator('text=/Initializing/i')
        expect(initializing_status).to_be_visible(timeout=10000)
        
        # Verify progress bar exists
        progress_bar = page.locator('progressbar')
        expect(progress_bar).to_be_visible()
        
        # Check for statistics section
        terms_processed = page.locator('text=Terms Processed')
        expect(terms_processed).to_be_visible()
        
        batches_complete = page.locator('text=Batches Complete')
        expect(batches_complete).to_be_visible()
        
        failed_terms = page.locator('text=Failed Terms')
        expect(failed_terms).to_be_visible()
        
        # Check for activity logs section
        activity_logs = page.locator('summary').filter(has_text='📋 Recent Activity Logs')
        expect(activity_logs).to_be_visible()
        
        # Expand logs to see details
        activity_logs.click()
        page.wait_for_timeout(2000)
        
        # Verify some initialization logs are present
        # Based on manual testing: logs show term parsing and initialization
        log_entry = page.locator('text=/Initializing embedding generation/i')
        expect(log_entry).to_be_visible(timeout=10000)
        
        # Verify cancel button is available
        cancel_btn = page.get_by_role('button', name='🛑 Cancel Generation')
        expect(cancel_btn).to_be_visible()
        
        # Cancel the generation to clean up
        cancel_btn.click()
        page.wait_for_timeout(2000)
        
        print("✅ Embeddings generation progress interface verified")
    
    def test_complete_end_to_end_workflow(self, page: Page):
        """Test the complete end-to-end workflow from download to embeddings start."""
        print("🚀 Starting complete end-to-end workflow test...")
        
        # Step 1: Download DOID
        print("Step 1: Downloading DOID...")
        page.get_by_role('button', name='📥 Ontology Updates').click()
        
        doid_expander = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        doid_expander.click()
        page.wait_for_timeout(1000)
        
        update_btn = page.get_by_role('button', name='🔄 Update from Source')
        update_btn.click()
        
        # Wait for download completion
        completion_msg = page.locator('text=/Update completed/i')
        expect(completion_msg).to_be_visible(timeout=60000)
        print("✅ DOID download completed")
        
        # Step 2: Navigate to Embeddings
        print("Step 2: Navigating to embeddings management...")
        page.get_by_role('button', name='🧠 Manage Embeddings').click()
        
        # Step 3: Verify downloads
        print("Step 3: Verifying downloads...")
        verify_btn = page.get_by_role('button', name='🔍 Verify Downloads')
        verify_btn.click()
        page.wait_for_timeout(10000)
        
        # Step 4: Check DOID is available
        print("Step 4: Checking DOID availability...")
        available_section = page.locator('text=/Available Ontologies for Embedding/i')
        expect(available_section).to_be_visible(timeout=15000)
        
        doid_embedding_section = page.locator('summary').filter(has_text='🧬 DOID - Disease Ontology')
        expect(doid_embedding_section).to_be_visible()
        print("✅ DOID is available for embeddings")
        
        # Step 5: Test embeddings interface
        print("Step 5: Testing embeddings interface...")
        doid_embedding_section.click()
        
        generate_btn = page.get_by_role('button', name='🚀 Generate Embeddings')
        expect(generate_btn).to_be_visible()
        print("✅ Generate embeddings button is available")
        
        print("🎉 Complete end-to-end workflow test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])