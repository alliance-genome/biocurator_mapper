"""End-to-end test for DO (Disease Ontology) embeddings functionality."""
from playwright.sync_api import sync_playwright
import time
import sys
import os

def test_do_embeddings():
    """Test complete flow: download DO ontology, then generate embeddings."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            # 1. Navigate and authenticate
            print("1. Navigating to app and authenticating...")
            page.goto("http://localhost:8501")
            page.wait_for_load_state("networkidle")
            
            # Enter API key
            page.fill('input[type="password"]', "1234")
            page.press('input[type="password"]', "Enter")
            time.sleep(2)
            
            # Check authentication
            page_text = page.text_content("body")
            if "🔧 Admin Panel" in page_text:
                print("   ✅ Admin features unlocked!")
            else:
                print("   ❌ Failed to unlock admin features")
                return False
            
            # 2. Download DO ontology first
            print("\n2. Downloading DO ontology...")
            
            # Navigate to Ontology Updates
            page.click('button:has-text("📥 Ontology Updates")')
            page.wait_for_timeout(2000)
            
            # Look for DOID section (might need to scroll or expand)
            print("   Looking for DOID section...")
            
            # Try to find and click DOID expander
            doid_header = page.locator('text=🧬 DOID - Disease Ontology').first
            if doid_header.is_visible():
                print("   Clicking DOID header to expand...")
                doid_header.click()
                time.sleep(1)
            
            # Find the Update button for DOID
            # Based on debug, button index 2 is visible and should be for DOID
            update_buttons = page.locator('button:has-text("🔄 Update from Source")')
            
            # Wait a bit for buttons to be ready
            page.wait_for_timeout(1000)
            
            # Count total buttons
            button_count = update_buttons.count()
            print(f"   Found {button_count} Update buttons total")
            
            # Try to click the visible button (should be index 2 based on debug)
            doid_update_clicked = False
            for i in range(button_count):
                btn = update_buttons.nth(i)
                if btn.is_visible():
                    print(f"   Found visible Update button at index {i}, clicking...")
                    btn.click()
                    doid_update_clicked = True
                    break
            
            if not doid_update_clicked:
                print("   ❌ Could not find visible DOID update button")
                page.screenshot(path="no_visible_update_button.png")
                return False
            
            # Wait for download to complete
            print("   Waiting for download to complete...")
            download_complete = False
            
            # First wait for any progress indication
            time.sleep(2)
            
            for i in range(30):  # Wait up to 30 seconds
                page_text = page.text_content("body")
                
                # Check for various completion indicators
                if any(indicator in page_text for indicator in [
                    "Download completed",
                    "Update completed", 
                    "completed successfully",
                    "✅ Update completed",
                    "100%"
                ]):
                    print(f"   ✅ DO download completed in {i+2}s")
                    download_complete = True
                    # Wait a bit more for UI to stabilize
                    time.sleep(2)
                    break
                    
                # Also check for download in progress
                if "Downloading" in page_text or "Update Progress" in page_text:
                    if i % 5 == 0:  # Print every 5 seconds
                        print(f"   ... Download in progress at {i+2}s")
                
                # For Streamlit, sometimes we need to look for the success message
                if "✅" in page_text and ("started" in page_text or "completed" in page_text):
                    # Take a screenshot to see what's happening
                    if i == 5:
                        page.screenshot(path="do_download_progress.png")
                
                time.sleep(1)
            
            if not download_complete:
                print("   ❌ Download did not complete in time")
                page.screenshot(path="do_download_timeout.png")
                return False
            
            # 3. Navigate to Embeddings page
            print("\n3. Navigating to Embeddings page...")
            
            # Click on Manage Embeddings button
            embeddings_btn = page.locator('button:has-text("🧠 Manage Embeddings")')
            if embeddings_btn.count() > 0:
                embeddings_btn.first.click()
                time.sleep(2)
            else:
                print("   ❌ Could not find Embeddings button")
                return False
            
            # Check if we're on embeddings page
            page_text = page.text_content("body")
            if "Ontology Embedding Management" not in page_text:
                print("   ❌ Not on embeddings page")
                page.screenshot(path="not_on_embeddings_page.png")
                return False
            
            print("   ✅ On Embeddings Management page")
            
            # 4. Generate embeddings for DO
            print("\n4. Generating embeddings for DO...")
            
            # Find DOID section on embeddings page
            doid_embedding_section = page.locator('text=🧬 DOID - Disease Ontology').first
            if doid_embedding_section.is_visible():
                print("   Clicking DOID section to expand...")
                doid_embedding_section.click()
                time.sleep(1)
            
            # Find Generate Embeddings button
            generate_btn = None
            doid_section = page.locator('div:has-text("Ontology: DOID")')
            if doid_section.count() > 0:
                generate_btn = doid_section.locator('button:has-text("🚀 Generate Embeddings")').first
                if generate_btn.is_visible():
                    print("   Clicking Generate Embeddings button...")
                    generate_btn.click()
                else:
                    print("   ❌ Generate Embeddings button not visible")
                    page.screenshot(path="no_generate_button.png")
                    return False
            
            # 5. Monitor embedding generation progress
            print("\n5. Monitoring embedding generation...")
            
            embedding_started = False
            embedding_completed = False
            error_found = False
            
            # Monitor for up to 5 minutes (DO is smaller than GO)
            for i in range(300):
                page_text = page.text_content("body")
                
                # Check for start
                if not embedding_started:
                    if any(text in page_text for text in [
                        "Starting embedding generation",
                        "Loading ontology data",
                        "Parsing ontology terms"
                    ]):
                        print(f"   ✅ Embedding generation started at {i}s")
                        embedding_started = True
                
                # Check for progress indicators
                if "Processing batch" in page_text:
                    # Extract batch info if possible
                    lines = page_text.split('\n')
                    for line in lines:
                        if "Processing batch" in line:
                            print(f"   📊 {line.strip()}")
                            break
                
                # Check for completion
                if "Embedding generation completed" in page_text or "Embeddings generated successfully" in page_text:
                    print(f"   ✅ Embedding generation completed at {i}s!")
                    embedding_completed = True
                    break
                
                # Check for errors
                if any(error in page_text for error in [
                    "Embedding generation failed",
                    "Error generating embeddings",
                    "Failed to generate embeddings"
                ]):
                    print(f"   ❌ Error detected at {i}s")
                    error_found = True
                    page.screenshot(path=f"embedding_error_{i}s.png")
                    break
                
                # Progress update every 10 seconds
                if i > 0 and i % 10 == 0:
                    print(f"   ... {i}s elapsed")
                
                time.sleep(1)
            
            # Take final screenshot
            page.screenshot(path="do_embeddings_final.png")
            
            # 6. Verify results
            print("\n6. Final results:")
            print(f"   Embedding started: {embedding_started}")
            print(f"   Embedding completed: {embedding_completed}")
            print(f"   Errors found: {error_found}")
            
            # Check final state
            final_text = page.text_content("body")
            
            # Look for success indicators
            if embedding_completed and not error_found:
                print("\n✅ TEST PASSED: DO embeddings generated successfully!")
                
                # Check if we can see collection info
                if "Collection:" in final_text:
                    lines = final_text.split('\n')
                    for line in lines:
                        if "Collection:" in line:
                            print(f"   {line.strip()}")
                
                return True
            else:
                print("\n❌ TEST FAILED: Embedding generation did not complete successfully")
                
                # Print recent logs if available
                if "Recent Logs" in final_text:
                    print("\n   Recent logs from page:")
                    lines = final_text.split('\n')
                    in_logs = False
                    for line in lines:
                        if "Recent Logs" in line:
                            in_logs = True
                        elif in_logs and line.strip():
                            print(f"     {line.strip()}")
                            if len([l for l in lines if l.strip()]) > 10:
                                break
                
                return False
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            try:
                page.screenshot(path="error_exception.png")
            except:
                pass
            return False
        finally:
            browser.close()


if __name__ == "__main__":
    # Make sure we have the OpenAI key for embeddings
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set in environment!")
        print("Embeddings will fail without it.")
        # Continue anyway to see what happens
    
    success = test_do_embeddings()
    sys.exit(0 if success else 1)