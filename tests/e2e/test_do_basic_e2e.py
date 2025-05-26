"""Basic e2e test to verify Playwright setup and UI interaction."""
import pytest
from playwright.sync_api import Page, expect


class TestBasicDOFunctionality:
    """Basic tests to verify Playwright and UI work correctly."""
    
    def test_ui_loads_and_authenticates(self, page: Page, base_url: str, admin_api_key: str):
        """Test that the UI loads and we can authenticate."""
        # Navigate to the app
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        # Check that the page loaded
        expect(page).to_have_title("BioCurator Mapper")
        
        # Enter API key
        api_input = page.locator('input[type="password"]')
        expect(api_input).to_be_visible(timeout=5000)
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        
        # Wait for authentication
        page.wait_for_timeout(2000)
        
        # Verify admin panel is visible
        admin_panel = page.locator("text=🔧 Admin Panel")
        expect(admin_panel).to_be_visible(timeout=10000)
        
        print("✅ UI loaded and authenticated successfully")
    
    def test_navigate_to_ontology_updates(self, page: Page, base_url: str, admin_api_key: str):
        """Test navigation to ontology updates page."""
        # Navigate and authenticate
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        
        # Wait for auth
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=10000)
        
        # Click Ontology Updates
        ontology_btn = page.locator('button:has-text("📥 Ontology Updates")')
        expect(ontology_btn).to_be_visible(timeout=5000)
        ontology_btn.click()
        
        # Verify navigation
        expect(page.locator('text=Ontology Update Management')).to_be_visible(timeout=5000)
        
        # Check for DOID section
        doid_section = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_section).to_be_visible(timeout=5000)
        
        print("✅ Successfully navigated to Ontology Updates and found DOID section")
    
    def test_navigate_to_embeddings_page(self, page: Page, base_url: str, admin_api_key: str):
        """Test navigation to embeddings management page."""
        # Navigate and authenticate
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        
        # Wait for auth
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=10000)
        
        # Click Manage Embeddings
        embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")')
        expect(embeddings_btn).to_be_visible(timeout=5000)
        embeddings_btn.click()
        
        # Verify navigation
        expect(page.locator("text=Ontology Embedding Management")).to_be_visible(timeout=5000)
        
        print("✅ Successfully navigated to Embeddings Management page")
    
    def test_do_section_expandable(self, page: Page, base_url: str, admin_api_key: str):
        """Test that DOID section is expandable in Ontology Updates."""
        # Navigate and authenticate
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        
        api_input = page.locator('input[type="password"]')
        api_input.fill(admin_api_key)
        api_input.press("Enter")
        
        # Wait for auth
        expect(page.locator("text=🔧 Admin Panel")).to_be_visible(timeout=10000)
        
        # Go to Ontology Updates
        page.locator('button:has-text("📥 Ontology Updates")').click()
        expect(page.locator('text=Ontology Update Management')).to_be_visible(timeout=5000)
        
        # Find and click DOID expander
        doid_expander = page.locator('text="🧬 DOID - Disease Ontology"')
        expect(doid_expander).to_be_visible(timeout=5000)
        
        # Click to expand
        doid_expander.click()
        page.wait_for_timeout(1000)
        
        # Check for Update button visibility
        update_btns = page.locator('button:has-text("🔄 Update from Source")')
        
        # At least one update button should be visible after expansion
        visible_count = 0
        for i in range(update_btns.count()):
            if update_btns.nth(i).is_visible():
                visible_count += 1
        
        assert visible_count > 0, "No Update buttons found after expanding DOID section"
        
        print(f"✅ DOID section expanded successfully, found {visible_count} Update button(s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])