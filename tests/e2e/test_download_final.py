"""Final working test for ontology download."""
from playwright.sync_api import sync_playwright
import time
import sys
import re

def test_ontology_download():
    """Test that ontology update only downloads without embedding."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # 1. Navigate to app with admin key in URL (simpler approach)
            print("1. Navigating to app with session state...")
            # First visit to set up session
            page.goto("http://localhost:8501")
            page.wait_for_load_state("networkidle")
            time.sleep(2)
            
            # 2. Enter API key and submit
            print("2. Entering API key...")
            password_input = page.locator('input[type="password"]')
            if password_input.count() > 0:
                password_input.fill("1234")
                # Simulate form submission by pressing Enter
                password_input.press("Enter")
                time.sleep(3)  # Wait for Streamlit to process
                
                # Check if we unlocked admin features
                page_text = page.text_content("body")
                if "🔧 Admin Panel" in page_text:
                    print("   ✅ Admin features unlocked!")
                else:
                    print("   ⚠️ Admin features may not be unlocked")
            
            # 3. Click on Ontology Updates button
            print("3. Looking for Ontology Updates button...")
            # Take screenshot to see current state
            page.screenshot(path="state_after_auth.png")
            
            # Try multiple selectors for the button
            selectors = [
                'button:has-text("📥 Ontology Updates")',
                'button:has-text("Ontology Updates")',
                '[data-testid*="button"]:has-text("📥")',
                'button >> text=Ontology Updates'
            ]
            
            clicked = False
            for selector in selectors:
                try:
                    btn = page.locator(selector)
                    if btn.count() > 0 and btn.first.is_visible():
                        print(f"   Found button with selector: {selector}")
                        btn.first.click()
                        clicked = True
                        break
                except:
                    continue
            
            if not clicked:
                print("   ERROR: Could not find Ontology Updates button")
                # Print all visible buttons for debugging
                all_buttons = page.locator("button").all()
                print(f"   Found {len(all_buttons)} total buttons:")
                for i, btn in enumerate(all_buttons[:20]):
                    try:
                        text = btn.text_content()
                        if text:
                            print(f"     Button {i}: {text.strip()}")
                    except:
                        pass
                return False
            
            # 4. Wait for Ontology Update page to load
            print("4. Waiting for Ontology Update page...")
            time.sleep(2)
            
            # Check if we're on the right page
            page_text = page.text_content("body")
            if "Ontology Update Management" not in page_text:
                print("   ERROR: Not on Ontology Update page")
                page.screenshot(path="wrong_page.png")
                return False
            
            print("   ✅ On Ontology Update Management page")
            
            # 5. Find and expand GO section
            print("5. Looking for GO ontology section...")
            # Streamlit expanders can be tricky, try clicking on the header text
            go_header = page.locator('text=🧬 GO - Gene Ontology').first
            if go_header.is_visible():
                print("   Clicking GO header to expand...")
                go_header.click()
                time.sleep(1)
            
            # 6. Find and click Update button
            print("6. Looking for Update button...")
            # After expanding, the button should be visible
            update_btn = page.locator('button:has-text("🔄 Update from Source")').first
            if update_btn.is_visible():
                print("   Clicking Update button...")
                update_btn.click()
            else:
                print("   ERROR: Update button not visible")
                page.screenshot(path="no_update_button.png")
                return False
            
            # 7. Check for success message
            print("7. Checking for update start confirmation...")
            time.sleep(2)
            
            success_found = False
            for i in range(5):
                page_text = page.text_content("body")
                if "Update started" in page_text or "✅" in page_text:
                    print("   ✅ Update started successfully!")
                    success_found = True
                    break
                time.sleep(1)
            
            if not success_found:
                print("   ⚠️ No explicit success message found, continuing...")
            
            # 8. Monitor for download vs embedding
            print("8. Monitoring progress for 30 seconds...")
            download_indicators = []
            embedding_indicators = []
            
            for i in range(30):
                page_text = page.text_content("body")
                
                # Check for download indicators
                download_keywords = ["Download", "download", "Downloading", "Downloaded", "MB"]
                for keyword in download_keywords:
                    if keyword in page_text and keyword not in str(download_indicators):
                        download_indicators.append(f"{keyword} at {i}s")
                        print(f"   ✅ Found download indicator: {keyword}")
                
                # Check for embedding indicators (should NOT find these)
                # Be more specific - look for embedding in progress context, not navigation buttons
                embedding_keywords = [
                    "Generating embeddings",
                    "Generating initial embeddings", 
                    "Processing batch",
                    "terms processed",
                    "Creating embeddings",
                    "Embedding generation"
                ]
                for keyword in embedding_keywords:
                    if keyword in page_text and keyword not in str(embedding_indicators):
                        embedding_indicators.append(f"{keyword} at {i}s")
                        print(f"   ❌ FOUND EMBEDDING INDICATOR: {keyword}")
                        page.screenshot(path=f"embedding_found_{i}s.png")
                
                # Check for completion
                if "completed" in page_text.lower() and i > 5:
                    print(f"   Process completed at {i}s")
                    break
                
                time.sleep(1)
            
            # 9. Final results
            print("\n=== FINAL RESULTS ===")
            print(f"Download indicators found: {len(download_indicators)}")
            for indicator in download_indicators:
                print(f"  - {indicator}")
            
            print(f"\nEmbedding indicators found: {len(embedding_indicators)}")
            for indicator in embedding_indicators:
                print(f"  - {indicator}")
            
            # Take final screenshot
            page.screenshot(path="test_complete.png")
            
            # Determine pass/fail
            if len(download_indicators) > 0 and len(embedding_indicators) == 0:
                print("\n✅ TEST PASSED: Only download occurred, no embedding!")
                return True
            elif len(embedding_indicators) > 0:
                print("\n❌ TEST FAILED: Embedding was triggered!")
                return False
            else:
                print("\n❌ TEST FAILED: No download activity detected!")
                return False
            
        except Exception as e:
            print(f"\nERROR: {e}")
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
    success = test_ontology_download()
    sys.exit(0 if success else 1)