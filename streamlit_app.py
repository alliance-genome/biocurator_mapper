import os
import requests
import streamlit as st
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

def perform_simple_download(ont_name: str, source_url: str) -> dict:
    """Perform a simple synchronous download."""
    result = {
        "success": False,
        "message": "",
        "file_path": None,
        "size_mb": 0,
        "duration_seconds": 0
    }
    
    start_time = time.time()
    
    try:
        # Create directory structure
        data_dir = Path("data/ontologies")
        ontology_dir = data_dir / ont_name
        ontology_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{ont_name.lower()}_{timestamp}.json"
        file_path = ontology_dir / filename
        
        # Download the file
        response = requests.get(source_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Write to file
        downloaded = 0
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        # Create/update the "latest" symlink
        latest_link = ontology_dir / f"{ont_name.lower()}_latest.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(filename)
        
        duration = time.time() - start_time
        size_mb = downloaded / (1024 * 1024)
        
        result.update({
            "success": True,
            "message": f"Successfully downloaded {size_mb:.2f} MB in {duration:.1f} seconds",
            "file_path": str(file_path),
            "size_mb": size_mb,
            "duration_seconds": duration
        })
        
        # Update version metadata
        update_version_metadata(ont_name, filename, size_mb)
        
    except requests.exceptions.Timeout:
        result["message"] = "Download timed out after 5 minutes"
    except requests.exceptions.RequestException as e:
        result["message"] = f"Download failed: {str(e)}"
    except Exception as e:
        result["message"] = f"Unexpected error: {str(e)}"
    
    return result


def update_version_metadata(ont_name: str, filename: str, size_mb: float):
    """Update the version metadata file."""
    metadata_file = Path("ontology_versions.json")
    
    try:
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        if ont_name not in metadata:
            metadata[ont_name] = []
        
        metadata[ont_name].append({
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat(),
            "size_mb": size_mb
        })
        
        # Keep only last 10 versions in metadata
        metadata[ont_name] = metadata[ont_name][-10:]
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        st.error(f"Failed to update version metadata: {e}")


def show_update_progress(ont_name: str, api_key: str):
    """Display ontology update progress."""
    st.markdown("---")
    st.markdown("**📥 Update Progress:**")
    
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/admin/update_progress/{ont_name}",
            headers={"X-API-Key": api_key},
            timeout=5
        )
        
        if resp.ok:
            progress_data = resp.json()
            status = progress_data.get("status", "unknown")
            percentage = progress_data.get("progress_percentage", 0)
            elapsed = progress_data.get("elapsed_seconds", 0)
            
            # Status message
            if status == "downloading":
                st.info(f"⬇️ Downloading ontology ({elapsed}s elapsed)...")
            elif status == "parsing":
                st.info(f"📝 Parsing ontology data ({elapsed}s elapsed)...")
            elif status == "version_check":
                st.info(f"🔍 Checking version ({elapsed}s elapsed)...")
            elif status == "embedding":
                st.info(f"🧠 Generating initial embeddings ({elapsed}s elapsed)...")
            elif status == "completed":
                st.success(f"✅ Update completed! ({elapsed}s total)")
            elif status == "failed":
                st.error(f"❌ Update failed ({elapsed}s elapsed)")
            
            # Progress bar
            st.progress(percentage / 100, text=f"Progress: {percentage}%")
            
            # Recent logs (no expander since we're already inside one)
            recent_logs = progress_data.get("recent_logs", [])
            if recent_logs:
                st.markdown("**📋 Recent Logs:**")
                log_text = ""
                for log in recent_logs[-5:]:  # Show last 5 logs
                    timestamp = log.get("timestamp", "")
                    message = log.get("message", "")
                    level = log.get("level", "INFO")
                    
                    if level == "ERROR":
                        log_text += f"🔴 [{timestamp}] {message}\n"
                    elif level == "WARNING":
                        log_text += f"🟡 [{timestamp}] {message}\n"
                    else:
                        log_text += f"⚪ [{timestamp}] {message}\n"
                
                st.text(log_text)
            
            # Clear button if completed/failed
            if status in ["completed", "failed"]:
                if st.button(f"🔄 Clear Status", key=f"clear_update_status_{ont_name}"):
                    del st.session_state[f"update_progress_{ont_name}"]
                    st.rerun()
            else:
                # Refresh button for ongoing updates
                if st.button(f"🔄 Refresh", key=f"refresh_update_{ont_name}"):
                    st.rerun()
        else:
            st.warning("⚠️ No active update found")
            del st.session_state[f"update_progress_{ont_name}"]
    
    except Exception as e:
        st.error(f"❌ Error checking progress: {str(e)}")

def get_download_history(ont_name: str) -> list:
    """Get download history for an ontology."""
    # Use the same path as the backend
    data_dir = Path(os.getenv("ONTOLOGY_DATA_DIR", "data"))
    metadata_file = data_dir / "ontology_downloads_history.json"
    
    if not metadata_file.exists():
        return []
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata.get(ont_name, [])
    except:
        return []


def clear_download_history(ont_name: str):
    """Clear download history for a specific ontology."""
    # Use the same path as the backend
    data_dir = Path(os.getenv("ONTOLOGY_DATA_DIR", "data"))
    metadata_file = data_dir / "ontology_downloads_history.json"
    
    if not metadata_file.exists():
        return
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if ont_name in metadata:
            metadata[ont_name] = []
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        st.error(f"Failed to clear download history: {e}")



def show_embedding_progress(ont_name: str, api_key: str):
    """Display embedding generation progress with real-time updates."""
    if f"embedding_progress_{ont_name}" not in st.session_state:
        return
    
    st.markdown("---")
    st.markdown("### 🤖 Embedding Progress")
    
    # Progress section is tracked via the containing elements
    
    # Get real-time progress from API
    try:
        resp = requests.get(
            f"{FASTAPI_URL}/admin/embedding_progress/{ont_name}",
            headers={"X-API-Key": api_key},
            timeout=5
        )
        
        if resp.ok:
            progress_data = resp.json()
            status = progress_data.get("status", "unknown")
            percentage = progress_data.get("progress_percentage", 0)
            elapsed = progress_data.get("elapsed_seconds", 0)
            
            # Create a container for status messages
            status_container = st.container()
            
            # Status message with more detail
            with status_container:
                # Status container will have test identifiers
                if status == "starting":
                    st.info(f"⏳ **Starting embedding generation**\n\nInitializing process... ({elapsed}s elapsed)")
                elif status == "initializing":
                    st.info(f"🔧 **Initializing**\n\nSetting up Weaviate collection and loading configuration... ({elapsed}s elapsed)")
                elif status == "loading":
                    st.info(f"📂 **Loading ontology data**\n\nReading ontology file from disk... ({elapsed}s elapsed)")
                elif status == "parsing":
                    st.info(f"🔍 **Parsing ontology terms**\n\nExtracting term data for embedding... ({elapsed}s elapsed)")
                elif status == "processing_terms":
                    st.info(f"📝 **Processing terms**\n\nPreparing terms for vectorization... ({elapsed}s elapsed)")
                elif status == "creating_collection":
                    st.info(f"🗄️ **Creating collection**\n\nSetting up Weaviate collection with OpenAI vectorizer... ({elapsed}s elapsed)")
                elif status == "embedding_generation" or status == "embedding_batch":
                    st.info(f"🧠 **Generating embeddings**\n\nCreating vector representations for terms... ({elapsed}s elapsed)")
                elif status == "retrying_batch":
                    st.warning(f"🔄 **Retrying failed batch**\n\nRetrying terms that failed in previous attempt... ({elapsed}s elapsed)")
                elif status == "rate_limited":
                    st.warning(f"⏸️ **Rate limited**\n\nWaiting before retry due to API rate limits... ({elapsed}s elapsed)")
                elif status == "finalizing":
                    st.info(f"📊 **Finalizing**\n\nUpdating configuration and cleaning up... ({elapsed}s elapsed)")
                elif status == "completed":
                    st.success(f"✅ **Embedding generation completed!**\n\nSuccessfully generated embeddings in {elapsed}s")
                elif status == "completed_with_errors":
                    st.warning(f"⚠️ **Completed with errors**\n\nEmbedding generation finished with some errors ({elapsed}s elapsed)")
                elif status == "completed_with_failures":
                    st.warning(f"⚠️ **Completed with failures**\n\nSome terms failed to embed ({elapsed}s elapsed)")
                elif status == "failed":
                    st.error(f"❌ **Embedding generation failed**\n\nProcess encountered an error ({elapsed}s elapsed)")
                elif status == "cancelled":
                    st.warning(f"⚠️ **Embedding generation cancelled**\n\nProcess was cancelled by user ({elapsed}s elapsed)")
                elif status == "cancelling":
                    st.warning(f"🛑 **Cancelling...**\n\nWaiting for safe cancellation point... ({elapsed}s elapsed)")
            
            # Enhanced progress bar
            progress_text = f"Progress: {percentage}%"
            if status == "embedding_batch" and embedding_stats:
                current_batch = embedding_stats.get("batches_completed", 0) + 1
                total_batches = embedding_stats.get("total_batches", 0)
                if total_batches > 0:
                    progress_text = f"Progress: {percentage}% (Batch {current_batch}/{total_batches})"
            
            # Add hidden div with data-testid for progress bar
            # Progress bar percentage tracked via the progress component
            st.progress(percentage / 100, text=progress_text)
            
            # Embedding statistics
            embedding_stats = progress_data.get("embedding_stats", {})
            if embedding_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Terms Processed", 
                             f"{embedding_stats.get('processed_terms', 0):,} / {embedding_stats.get('total_terms', 0):,}")
                with col2:
                    st.metric("Batches Complete", 
                             f"{embedding_stats.get('batches_completed', 0)} / {embedding_stats.get('total_batches', 0)}")
                with col3:
                    terms_per_sec = progress_data.get("terms_per_second", 0)
                    if terms_per_sec > 0:
                        st.metric("Speed", f"{terms_per_sec:.1f} terms/s")
                    else:
                        st.metric("Failed Terms", embedding_stats.get('failed_terms', 0))
            
            # Recent logs in an expandable section
            recent_logs = progress_data.get("recent_logs", [])
            if recent_logs:
                with st.expander("📋 Recent Activity Logs", 
                                expanded=(status == "failed")):
                    for log in reversed(recent_logs):  # Show newest first
                        timestamp = log.get("timestamp", "")
                        message = log.get("message", "")
                        level = log.get("level", "INFO")
                        
                        # Color-code by level
                        if level == "ERROR":
                            st.error(f"🔴 [{timestamp}] {message}")
                        elif level == "WARNING":
                            st.warning(f"🟡 [{timestamp}] {message}")
                        else:
                            st.info(f"⚪ [{timestamp}] {message}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Cancel button for embeddings (only if not completed)
                if status not in ["completed", "failed", "cancelled", "completed_with_errors", "completed_with_failures"]:
                    # Add hidden div with data-testid for cancel button
                    # Cancel button has its own test identifier
                    if st.button(f"🛑 Cancel Generation", key=f"cancel_embedding_{ont_name}", type="secondary", use_container_width=True):
                        with st.spinner("Sending cancellation request..."):
                            try:
                                cancel_resp = requests.delete(
                                    f"{FASTAPI_URL}/admin/embedding_progress/{ont_name}",
                                    headers={"X-API-Key": api_key},
                                    timeout=5
                                )
                                if cancel_resp.ok:
                                    st.success("✅ Cancellation requested. The process will stop at the next safe point.")
                                    time.sleep(1)  # Brief pause to show message
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to cancel embedding generation")
                            except Exception as e:
                                st.error(f"❌ Error cancelling: {str(e)}")
            
            with col2:
                # Clear button if completed/failed/cancelled
                if status in ["completed", "failed", "cancelled", "completed_with_errors", "completed_with_failures"]:
                    if st.button(f"✓ Clear Status", key=f"clear_embedding_{ont_name}", type="primary", use_container_width=True):
                        del st.session_state[f"embedding_progress_{ont_name}"]
                        st.rerun()
        
        else:
            # No active embedding generation found
            st.warning("⚠️ No active embedding generation found")
            del st.session_state[f"embedding_progress_{ont_name}"]
            
    except Exception as e:
        st.error(f"❌ Error checking embedding progress: {str(e)}")
        # Keep the progress state to retry

@st.cache_data
def load_ontology_config():
    """Load ontology configuration from YAML file."""
    config_path = os.getenv("ONTOLOGY_CONFIG_PATH", "ontology_config.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "ontologies": {
                "GO": {"name": "Gene Ontology", "enabled": True},
                "DOID": {"name": "Disease Ontology", "enabled": True}
            }
        }

@st.cache_data
def load_embeddings_config():
    """Load embeddings configuration from YAML file."""
    config_path = os.getenv("EMBEDDINGS_CONFIG_PATH", "embeddings_config.yaml")
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default configuration if file doesn't exist
        return {
            "model": {
                "name": "text-ada-002",
                "dimensions": 1536
            },
            "processing": {
                "batch_size": 100,
                "parallel_processing": True,
                "retry_failed": True,
                "max_retries": 3
            },
            "vectorize_fields": {
                "name": True,
                "definition": True,
                "synonyms": True
            },
            "preprocessing": {
                "lowercase": False,
                "remove_punctuation": False,
                "combine_fields_separator": " | "
            },
            "performance": {
                "request_timeout": 30,
                "rate_limit_delay": 0.1
            },
            "usage": {
                "track_tokens": True,
                "log_requests": False
            }
        }

def save_embeddings_config(config):
    """Save embeddings configuration to YAML file."""
    config_path = os.getenv("EMBEDDINGS_CONFIG_PATH", "embeddings_config.yaml")
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        # Clear the cached config data
        load_embeddings_config.clear()
        return True
    except Exception as e:
        st.error(f"Failed to save embeddings configuration: {str(e)}")
        return False

def get_enabled_ontologies():
    """Get list of enabled ontology names."""
    config = load_ontology_config()
    return [name for name, config_data in config.get("ontologies", {}).items() 
            if config_data.get("enabled", True)]

def format_ontology_term(term: Dict[str, Any]) -> str:
    """Format an ontology term for display."""
    if not term:
        return "None"
    
    name = term.get('name', 'Unknown')
    term_id = term.get('id', 'Unknown')
    definition = term.get('definition', '')
    
    formatted = f"**{name}** (`{term_id}`)"
    if definition:
        formatted += f"\n\n*Definition:* {definition}"
    
    return formatted

def check_system_health(api_key: str) -> Dict[str, Any]:
    """Check system health including Weaviate and OpenAI API status."""
    health_status = {
        "fastapi": {"status": "unknown", "details": ""},
        "weaviate": {"status": "unknown", "details": ""},
        "openai": {"status": "unknown", "details": ""},
        "overall": "unknown"
    }
    
    try:
        # Check FastAPI health
        resp = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if resp.ok:
            health_status["fastapi"]["status"] = "healthy"
            health_status["fastapi"]["details"] = "FastAPI service is responding"
        else:
            health_status["fastapi"]["status"] = "unhealthy"
            health_status["fastapi"]["details"] = f"HTTP {resp.status_code}"
    except Exception as e:
        health_status["fastapi"]["status"] = "error"
        health_status["fastapi"]["details"] = str(e)
    
    try:
        # Check Weaviate status via admin endpoint
        resp = requests.get(
            f"{FASTAPI_URL}/admin/weaviate_health",
            headers={"X-API-Key": api_key},
            timeout=10
        )
        if resp.ok:
            weaviate_data = resp.json()
            health_status["weaviate"]["status"] = "healthy" if weaviate_data.get("healthy") else "unhealthy"
            health_status["weaviate"]["details"] = weaviate_data.get("details", "Connected")
        else:
            health_status["weaviate"]["status"] = "unhealthy"
            health_status["weaviate"]["details"] = f"HTTP {resp.status_code}"
    except Exception as e:
        health_status["weaviate"]["status"] = "error"
        health_status["weaviate"]["details"] = str(e)
    
    try:
        # Check OpenAI API status via admin endpoint
        resp = requests.get(
            f"{FASTAPI_URL}/admin/openai_health",
            headers={"X-API-Key": api_key},
            timeout=10
        )
        if resp.ok:
            openai_data = resp.json()
            health_status["openai"]["status"] = "healthy" if openai_data.get("healthy") else "unhealthy"
            health_status["openai"]["details"] = openai_data.get("details", "API key valid")
        else:
            health_status["openai"]["status"] = "unhealthy"
            health_status["openai"]["details"] = f"HTTP {resp.status_code}"
    except Exception as e:
        health_status["openai"]["status"] = "error"
        health_status["openai"]["details"] = str(e)
    
    # Determine overall health
    statuses = [health_status[service]["status"] for service in ["fastapi", "weaviate", "openai"]]
    if all(status == "healthy" for status in statuses):
        health_status["overall"] = "healthy"
    elif any(status == "error" for status in statuses):
        health_status["overall"] = "error"
    else:
        health_status["overall"] = "degraded"
    
    return health_status

def display_health_status(health_status: Dict[str, Any]):
    """Display system health status with colored indicators."""
    overall_status = health_status["overall"]
    
    if overall_status == "healthy":
        st.success("🟢 System Healthy")
    elif overall_status == "degraded":
        st.warning("🟡 System Degraded")
    else:
        st.error("🔴 System Issues Detected")
    
    # Display service details directly without an expander
    st.markdown("**Service Details:**")
    for service, status_info in health_status.items():
        if service == "overall":
            continue
        
        status = status_info["status"]
        details = status_info["details"]
        
        if status == "healthy":
            st.success(f"✅ {service.upper()}: {details}")
        elif status == "unhealthy":
            st.warning(f"⚠️ {service.upper()}: {details}")
        else:
            st.error(f"❌ {service.upper()}: {details}")

def check_openai_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    # Check environment variables that might contain the OpenAI API key
    openai_key_vars = [
        "OPENAI_API_KEY",
        "OPENAI_KEY", 
        "OPENAI_TOKEN"
    ]
    
    # Known invalid/placeholder values
    invalid_values = {
        "none", "null", "", "your_api_key_here", "sk-", 
        "your_openai_api_key", "change_me", "placeholder"
    }
    
    for var in openai_key_vars:
        key = os.getenv(var)
        if key and key.strip():
            key_clean = key.strip().lower()
            # Check if it's not in invalid values and has reasonable length
            if key_clean not in invalid_values and len(key_clean) > 10:
                return True
    
    return False

def display_openai_warning():
    """Display warning if OpenAI API key is not configured."""
    if not check_openai_api_key():
        st.warning(
            "⚠️ **OpenAI API Key Not Detected** - "
            "The system requires an OpenAI API key to function properly. "
            "Please set the `OPENAI_API_KEY` environment variable. "
            "Without it, ontology term resolution will fail."
        )
        
        with st.expander("ℹ️ How to configure OpenAI API Key"):
            st.markdown("""
            **To configure your OpenAI API key:**
            
            1. **Get an API key** from [OpenAI Platform](https://platform.openai.com/account/api-keys)
            
            2. **Set the environment variable** in one of these ways:
            
            **Option A: Docker Environment**
            ```bash
            # In your docker-compose.yml or when running containers
            environment:
              - OPENAI_API_KEY=your_actual_api_key_here
            ```
            
            **Option B: Local Development**
            ```bash
            # In your terminal or .env file
            export OPENAI_API_KEY=your_actual_api_key_here
            ```
            
            **Option C: System Environment Variable**
            ```bash
            # Add to your shell profile (.bashrc, .zshrc, etc.)
            echo 'export OPENAI_API_KEY=your_actual_api_key_here' >> ~/.bashrc
            source ~/.bashrc
            ```
            
            3. **Restart the application** after setting the environment variable
            
            **Note:** Keep your API key secure and never commit it to version control!
            """)
        
        return True
    return False

# Main UI
st.title("Biocurator Mapper")
st.write("Resolve passages to ontology terms using LLM assistance")

# Check for OpenAI API key and display warning if missing
display_openai_warning()

# Display health status prominently at the top if it was just checked
if hasattr(st.session_state, 'health_status') and st.session_state.get('show_health_dashboard', False):
    st.markdown("---")
    st.markdown("## 🏥 System Health Dashboard")
    display_health_status(st.session_state.health_status)
    st.markdown("---")

# Ontology Config Editor
if st.session_state.get('show_config_editor', False):
    st.markdown("---")
    st.markdown("## ⚙️ Ontology Configuration Editor")
    
    # Load current config
    try:
        config_path = os.getenv("ONTOLOGY_CONFIG_PATH", "ontology_config.yaml")
        with open(config_path, 'r') as f:
            config_text = f.read()
            if 'config_editor_text' not in st.session_state:
                st.session_state.config_editor_text = config_text
                st.session_state.original_config_text = config_text
    except Exception as e:
        st.error(f"Failed to load configuration: {str(e)}")
        config_text = ""
    
    # Initialize editor version counter if not exists
    if 'editor_version' not in st.session_state:
        st.session_state.editor_version = 0
    
    # YAML editor with dynamic key
    edited_config = st.text_area(
        "Edit YAML Configuration",
        value=st.session_state.get('config_editor_text', config_text),
        height=400,
        key=f"yaml_editor_{st.session_state.editor_version}"
    )
    
    # Update session state
    st.session_state.config_editor_text = edited_config
    
    # Validation
    config_valid = True
    error_message = ""
    try:
        yaml.safe_load(edited_config)
        st.success("✅ Valid YAML syntax")
    except yaml.YAMLError as e:
        config_valid = False
        error_message = str(e)
        st.error(f"❌ Invalid YAML: {error_message}")
    
    # Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save", type="primary", disabled=not config_valid):
            try:
                # Save to file
                with open(config_path, 'w') as f:
                    f.write(edited_config)
                
                # Call API to reload config
                resp = requests.post(
                    f"{FASTAPI_URL}/admin/reload_config",
                    headers={"X-API-Key": st.session_state.get('api_key', '')}
                )
                
                if resp.ok:
                    st.success("✅ Configuration saved and reloaded!")
                    st.session_state.original_config_text = edited_config
                    # Clear the cached config data
                    load_ontology_config.clear()
                    # Don't close the editor - let user continue editing or close manually
                    time.sleep(1)
                else:
                    st.error(f"❌ Failed to reload config: {resp.text}")
            except Exception as e:
                st.error(f"❌ Failed to save: {str(e)}")
    
    with col2:
        if st.button("↩️ Revert"):
            # Reload the original content from file
            try:
                with open(config_path, 'r') as f:
                    original_content = f.read()
                st.session_state.config_editor_text = original_content
                st.session_state.original_config_text = original_content
                # Increment editor version to force new widget
                st.session_state.editor_version += 1
                # Clear the cached config data
                load_ontology_config.clear()
            except Exception as e:
                st.error(f"Failed to reload file: {str(e)}")
            st.rerun()
    
    with col3:
        if st.button("❌ Cancel"):
            st.session_state.show_config_editor = False
            if 'config_editor_text' in st.session_state:
                del st.session_state.config_editor_text
            if 'original_config_text' in st.session_state:
                del st.session_state.original_config_text
            if 'editor_version' in st.session_state:
                del st.session_state.editor_version
            st.rerun()
    
    st.markdown("---")

# Embeddings Configuration Editor
if st.session_state.get('show_embeddings_config', False):
    st.markdown("---")
    st.markdown("## 🧠 Embeddings Configuration")
    
    # Get current embedding config (hardcoded defaults for now)
    embedding_models = {
        "text-ada-002": "OpenAI Ada v2",
        "text-embedding-3-small": "OpenAI Embedding v3 Small", 
        "text-embedding-3-large": "OpenAI Embedding v3 Large"
    }
    
    model_descriptions = {
        "text-ada-002": "Fastest and most economical option. Good for most use cases.",
        "text-embedding-3-small": "Balanced performance and cost. Better quality than Ada.",
        "text-embedding-3-large": "Highest quality embeddings. Best for complex matching."
    }
    
    model_dimensions = {
        "text-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072
    }
    
    # Load current embeddings configuration from YAML file
    embeddings_config = load_embeddings_config()
    
    # Initialize session state for embedding config if not exists
    if 'embedding_config' not in st.session_state:
        st.session_state.embedding_config = embeddings_config.copy()
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Model Configuration")
        
        # Model selection
        current_model = st.session_state.embedding_config.get("model", {}).get("name", "text-ada-002")
        # Add hidden div with data-testid for model selectbox
        # Model selection test identifier on the selectbox
        selected_model = st.selectbox(
            "Embedding Model",
            options=list(embedding_models.keys()),
            index=list(embedding_models.keys()).index(current_model) if current_model in embedding_models else 0,
            format_func=lambda x: embedding_models[x],
            help="Select the OpenAI embedding model to use for vectorizing ontology terms",
            key="select-embedding-model"
        )
        
        # Show model info
        st.info(f"📊 **Dimensions:** {model_dimensions[selected_model]}")
        st.markdown(f"**Description:** {model_descriptions[selected_model]}")
        
        # Batch size configuration
        current_batch_size = st.session_state.embedding_config.get("processing", {}).get("batch_size", 100)
        # Add hidden div with data-testid for batch size input
        # Batch size input test identifier on the number input
        batch_size = st.number_input(
            "Batch Size",
            min_value=10,
            max_value=1000,
            value=current_batch_size,
            step=10,
            help="Number of terms to process in each batch during ontology updates",
            key="input-batch-size"
        )
        
        # Cost estimation (approximate)
        cost_per_1k = {"text-ada-002": 0.0001, "text-embedding-3-small": 0.00002, "text-embedding-3-large": 0.00013}
        st.markdown(f"💰 **Estimated Cost:** ~${cost_per_1k[selected_model]:.5f} per 1K tokens")
    
    with col2:
        st.markdown("### ⚙️ Vectorization Settings")
        
        # Vectorization fields
        st.markdown("**Fields to Vectorize:**")
        current_vectorize = st.session_state.embedding_config.get("vectorize_fields", {})
        vectorize_name = st.checkbox(
            "Term Name", 
            value=current_vectorize.get("name", True),
            help="Include the ontology term name in the embedding"
        )
        vectorize_definition = st.checkbox(
            "Definition", 
            value=current_vectorize.get("definition", True),
            help="Include the term definition in the embedding"
        )
        vectorize_synonyms = st.checkbox(
            "Synonyms", 
            value=current_vectorize.get("synonyms", True),
            help="Include term synonyms in the embedding"
        )
        
        st.markdown("**Text Preprocessing:**")
        current_preprocessing = st.session_state.embedding_config.get("preprocessing", {})
        lowercase_text = st.checkbox(
            "Convert to Lowercase", 
            value=current_preprocessing.get("lowercase", False),
            help="Convert all text to lowercase before embedding"
        )
        remove_punctuation = st.checkbox(
            "Remove Punctuation", 
            value=current_preprocessing.get("remove_punctuation", False),
            help="Remove punctuation marks from text before embedding"
        )
        
        # Performance settings
        st.markdown("**Performance:**")
        current_processing = st.session_state.embedding_config.get("processing", {})
        parallel_processing = st.checkbox(
            "Enable Parallel Processing", 
            value=current_processing.get("parallel_processing", True),
            help="Process multiple batches concurrently"
        )
        retry_failed = st.checkbox(
            "Retry Failed Embeddings", 
            value=current_processing.get("retry_failed", True),
            help="Automatically retry failed embedding requests"
        )
    
    # Update session state with proper YAML structure
    st.session_state.embedding_config.update({
        "model": {
            "name": selected_model,
            "dimensions": model_dimensions[selected_model]
        },
        "processing": {
            "batch_size": batch_size,
            "parallel_processing": parallel_processing,
            "retry_failed": retry_failed,
            "max_retries": current_processing.get("max_retries", 3)
        },
        "vectorize_fields": {
            "name": vectorize_name,
            "definition": vectorize_definition,
            "synonyms": vectorize_synonyms
        },
        "preprocessing": {
            "lowercase": lowercase_text,
            "remove_punctuation": remove_punctuation,
            "combine_fields_separator": current_preprocessing.get("combine_fields_separator", " | ")
        },
        "performance": embeddings_config.get("performance", {
            "request_timeout": 30,
            "rate_limit_delay": 0.1
        }),
        "usage": embeddings_config.get("usage", {
            "track_tokens": True,
            "log_requests": False
        })
    })
    
    # Configuration preview
    st.markdown("### 📋 Configuration Preview")
    with st.expander("View Full Configuration", expanded=False):
        st.json(st.session_state.embedding_config)
    
    # Action buttons
    st.markdown("### 💾 Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Add hidden div with data-testid for save button
        # Save button test identifier on the button
        if st.button("💾 Save Configuration", type="primary", key="button-save-embedding-config"):
            if save_embeddings_config(st.session_state.embedding_config):
                st.success("✅ Configuration saved to embeddings_config.yaml!")
                # Update the session state to reflect saved changes
                st.session_state.embedding_config = load_embeddings_config().copy()
            else:
                st.error("❌ Failed to save configuration")
    
    with col2:
        # Add hidden div with data-testid for test config button
        # Test button test identifier on the button
        if st.button("🔄 Test Configuration", key="button-test-embedding-config"):
            with st.spinner("Testing embedding configuration..."):
                try:
                    api_key = st.session_state.get('api_key', '')
                    if not api_key:
                        st.error("❌ API key required for testing")
                    else:
                        # Call API endpoint to test embedding
                        resp = requests.post(
                            f"{FASTAPI_URL}/admin/test_embeddings_config",
                            headers={"X-API-Key": api_key},
                            timeout=10
                        )
                        
                        if resp.ok:
                            data = resp.json()
                            if data.get("success"):
                                st.success("✅ Configuration test successful!")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Model", data.get("model", "Unknown"))
                                    st.metric("Dimensions", data.get("dimensions", 0))
                                with col_b:
                                    st.info(f"Test text: \"{data.get('test_text', '')}\"")
                                    preview = data.get("embedding_preview", [])
                                    if preview:
                                        st.code(f"Embedding preview: {preview[:3]}...")
                            else:
                                error = data.get("error", "unknown")
                                message = data.get("message", "Configuration test failed")
                                if error == "authentication_error":
                                    st.error(f"🔑 {message}")
                                elif error == "model_not_found":
                                    st.error(f"🔍 {message}")
                                elif error == "rate_limit":
                                    st.warning(f"⏱️ {message}")
                                else:
                                    st.error(f"❌ {message}")
                        else:
                            st.error(f"❌ API request failed: {resp.text}")
                except Exception as e:
                    st.error(f"❌ Error testing configuration: {str(e)}")
    
    with col3:
        if st.button("↩️ Reset to Defaults"):
            # Reset to the values currently saved in the YAML file
            st.session_state.embedding_config = load_embeddings_config().copy()
            st.success("✅ Reset to saved configuration")
            st.rerun()
    
    with col4:
        if st.button("❌ Cancel"):
            st.session_state.show_embeddings_config = False
            if 'embedding_config' in st.session_state:
                del st.session_state.embedding_config
            st.rerun()
    
    # Simple warning about re-embedding
    st.info(
        "💡 **Note:** After changing embedding settings, you'll need to re-embed ontologies using the "
        "**Ontology Management** page to apply the new configuration."
    )
    
    st.markdown("---")

# Ontology Update Management Page
if st.session_state.get('show_ontology_update_management', False):
    st.markdown("---")
    st.markdown("## 📥 Ontology Update Management")
    st.markdown("Manage ontology downloads, updates, and version control")
    
    # Load ontology configuration
    config = load_ontology_config()
    ontologies = config.get("ontologies", {})
    
    if not ontologies:
        st.warning("⚠️ No ontologies configured. Please check your ontology configuration.")
    else:
        # Filter enabled ontologies
        enabled_ontologies_list = [(name, conf) for name, conf in ontologies.items() if conf.get("enabled", True)]
        
        if not enabled_ontologies_list:
            st.info("No enabled ontologies found.")
        else:
            st.markdown("### 📊 Available Ontologies")
            
            # Create columns for better layout
            for ont_name, ont_config in enabled_ontologies_list:
                # Check if this ontology has active update progress
                has_active_update = f"update_progress_{ont_name}" in st.session_state
                
                with st.expander(f"🧬 {ont_name} - {ont_config.get('name', 'Unknown')}", 
                                expanded=has_active_update):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Ontology:** {ont_name}")
                        st.markdown(f"**Full Name:** {ont_config.get('name', 'Unknown')}")
                        
                        # Source URL input
                        default_url = ont_config.get("default_source_url", "")
                        source_url = st.text_input(
                            "Source URL", 
                            value=default_url, 
                            key=f"update_url_{ont_name}",
                            help="URL to download the ontology data from"
                        )
                        
                        # Show download history
                        download_history = get_download_history(ont_name)
                        if download_history:
                            st.markdown("#### 📜 Version History")
                            history_df_data = []
                            for version in download_history[-5:]:  # Show last 5
                                timestamp = datetime.fromisoformat(version['timestamp'])
                                history_df_data.append({
                                    "Date": timestamp.strftime('%Y-%m-%d %H:%M'),
                                    "File": version['filename'],
                                    "Size (MB)": f"{version['size_mb']:.2f}"
                                })
                            if history_df_data:
                                import pandas as pd
                                st.dataframe(pd.DataFrame(history_df_data), use_container_width=True, hide_index=True)
                            
                            if st.button("🗑️ Clear History", key=f"clear_update_history_{ont_name}"):
                                clear_download_history(ont_name)
                                st.success("History cleared")
                                st.rerun()
                    
                    with col2:
                        st.markdown("**Actions:**")
                        
                        # Update button
                        if st.button(
                            "🔄 Update from Source", 
                            key=f"update_btn_{ont_name}",
                            type="primary",
                            help="Download latest version and check for updates"
                        ):
                            if not source_url.strip():
                                st.error("❌ Source URL is required")
                            else:
                                # Start update process
                                with st.spinner("Starting update..."):
                                    try:
                                        api_key = st.session_state.get('api_key', '')
                                        if not api_key:
                                            st.error("❌ API key required")
                                            st.stop()
                                        
                                        resp = requests.post(
                                            f"{FASTAPI_URL}/admin/update_ontology",
                                            headers={"X-API-Key": api_key},
                                            json={
                                                "ontology_name": ont_name,
                                                "source_url": source_url
                                            },
                                            timeout=10
                                        )
                                        
                                        if resp.ok:
                                            st.success("✅ Update started")
                                            st.session_state[f"update_progress_{ont_name}"] = {
                                                "status": "started",
                                                "timestamp": time.time()
                                            }
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed: {resp.text}")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
            
            # Show any active update progress AFTER all expanders
            st.markdown("---")
            for ont_name, ont_config in enabled_ontologies_list:
                if f"update_progress_{ont_name}" in st.session_state:
                    show_update_progress(ont_name, st.session_state.get('api_key', ''))
    
    # Back button
    st.markdown("---")
    if st.button("❌ Back to Home", key="back_from_update_mgmt"):
        st.session_state.show_ontology_update_management = False
        # Clear update-related states
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('update_progress_', 'update_url_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

# Ontology Embedding Management Page  
if st.session_state.get('show_ontology_embedding_management', False):
    st.markdown("---")
    st.markdown("## 🧠 Ontology Embedding Management")
    st.markdown("Generate and manage embeddings for downloaded ontologies")
    
    # Load configurations
    config = load_ontology_config()
    embeddings_config = load_embeddings_config()
    ontologies = config.get("ontologies", {})
    
    # Display current embedding configuration in a nice box
    model_name = embeddings_config.get("model", {}).get("name", "text-ada-002")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"📊 **Current Configuration**\n\n"
                    f"**Model:** {model_name}\n\n"
                    f"**Batch Size:** {embeddings_config.get('processing', {}).get('batch_size', 100)}\n\n"
                    f"**Fields:** {', '.join([k for k, v in embeddings_config.get('vectorize_fields', {}).items() if v])}")
        with col2:
            if st.button("🔍 Test Config", key="test_embed_config_quick", use_container_width=True, help="Test the current embedding configuration"):
                with st.spinner("Testing..."):
                    try:
                        api_key = st.session_state.get('api_key', '')
                        if api_key:
                            resp = requests.post(
                                f"{FASTAPI_URL}/admin/test_embeddings_config",
                                headers={"X-API-Key": api_key},
                                timeout=5
                            )
                            if resp.ok and resp.json().get("success"):
                                st.success("✅ Config OK!")
                            else:
                                st.error("❌ Config issue")
                        else:
                            st.error("🔑 API key required")
                    except:
                        st.error("❌ Test failed")
        with col3:
            if st.button("⚙️ Configure", key="link_to_embed_config", use_container_width=True):
                st.session_state.show_embeddings_config = True
                st.session_state.show_ontology_embedding_management = False
                st.rerun()
    
    st.markdown("---")
    
    if not ontologies:
        st.warning("⚠️ No ontologies configured.")
    else:
        # Check which ontologies have been downloaded (exist in ontology_versions.json)
        available_for_embedding = []
        for ont_name, ont_config in ontologies.items():
            if ont_config.get("enabled", True):
                # Check if ontology has been downloaded
                history = get_download_history(ont_name)
                if history:  # Has download history = available for embedding
                    available_for_embedding.append((ont_name, ont_config))
        
        if not available_for_embedding:
            st.warning("⚠️ No ontologies available for embedding. Please download ontologies first using the Update Management page.")
        else:
            st.markdown("### 📋 Available Ontologies for Embedding")
            
            # Cost estimation
            cost_per_1k = {
                "text-ada-002": 0.0001,
                "text-embedding-3-small": 0.00002,
                "text-embedding-3-large": 0.00013
            }
            estimated_cost = cost_per_1k.get(model_name, 0.0001)
            
            for ont_name, ont_config in available_for_embedding:
                # Check if embedding is in progress
                has_active_embedding = f"embedding_progress_{ont_name}" in st.session_state
                
                # Expander with key for test identifier
                with st.expander(f"🧬 {ont_name} - {ont_config.get('name', 'Unknown')}", 
                                expanded=has_active_embedding):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Ontology:** {ont_name}")
                        st.markdown(f"**Full Name:** {ont_config.get('name', 'Unknown')}")
                        
                        # Show last download info
                        history = get_download_history(ont_name)
                        if history:
                            last_version = history[-1]
                            timestamp = datetime.fromisoformat(last_version['timestamp'])
                            st.markdown(f"**Last Updated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            st.markdown(f"**Current Version:** {last_version['filename']}")
                        
                        # Embedding info
                        st.markdown(f"**Embedding Model:** {model_name}")
                        st.markdown(f"💰 **Estimated Cost:** ~${estimated_cost:.5f} per 1,000 tokens")
                        
                        # Vectorization settings summary
                        st.markdown("**Vectorization Settings:**")
                        vectorize_fields = embeddings_config.get("vectorize_fields", {})
                        fields = []
                        if vectorize_fields.get("name", True): fields.append("Name")
                        if vectorize_fields.get("definition", True): fields.append("Definition")
                        if vectorize_fields.get("synonyms", True): fields.append("Synonyms")
                        st.markdown(f"- Fields: {', '.join(fields)}")
                        st.markdown(f"- Batch Size: {embeddings_config.get('processing', {}).get('batch_size', 100)}")
                    
                    with col2:
                        st.markdown("**Actions:**")
                        
                        # Generate embeddings button with confirmation
                        if f"confirm_embed_{ont_name}" not in st.session_state:
                            # Generate button will have test identifier
                            if st.button(
                                "🚀 Generate Embeddings",
                                key=f"gen_embed_btn_{ont_name}",
                                type="primary",
                                help="Generate embeddings for this ontology"
                            ):
                                st.session_state[f"confirm_embed_{ont_name}"] = True
                                st.rerun()
                        else:
                            # Show confirmation
                            st.error("⚠️ **This will incur costs!**")
                            st.warning(f"Generate embeddings for {ont_name}?")
                            st.info(f"Est. cost: ~${estimated_cost:.5f}/1K tokens")
                            
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Confirm", key=f"confirm_yes_{ont_name}"):
                                    # Start embedding generation
                                    with st.spinner("Starting..."):
                                        try:
                                            api_key = st.session_state.get('api_key', '')
                                            if not api_key:
                                                st.error("❌ API key required")
                                                st.stop()
                                            
                                            # Check OpenAI health first
                                            health_resp = requests.get(
                                                f"{FASTAPI_URL}/admin/openai_health",
                                                headers={"X-API-Key": api_key},
                                                timeout=5
                                            )
                                            
                                            if health_resp.ok:
                                                health_data = health_resp.json()
                                                if not health_data.get("healthy", False):
                                                    st.error(f"❌ OpenAI API issue: {health_data.get('details', 'Unknown')}")
                                                    del st.session_state[f"confirm_embed_{ont_name}"]
                                                    st.stop()
                                            
                                            # Start embeddings
                                            resp = requests.post(
                                                f"{FASTAPI_URL}/admin/generate_embeddings",
                                                headers={"X-API-Key": api_key},
                                                json={"ontology_name": ont_name},
                                                timeout=10
                                            )
                                            
                                            if resp.ok:
                                                st.success("✅ Started")
                                                st.session_state[f"embedding_progress_{ont_name}"] = {
                                                    "status": "started",
                                                    "timestamp": time.time()
                                                }
                                                del st.session_state[f"confirm_embed_{ont_name}"]
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed: {resp.text}")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                                    del st.session_state[f"confirm_embed_{ont_name}"]
                            
                            with col_no:
                                if st.button("❌ Cancel", key=f"confirm_no_{ont_name}"):
                                    del st.session_state[f"confirm_embed_{ont_name}"]
                                    st.rerun()
                
                # Show embedding progress OUTSIDE the expander
                if f"embedding_progress_{ont_name}" in st.session_state:
                    show_embedding_progress(ont_name, st.session_state.get('api_key', ''))
    
    # Note about embeddings
    st.markdown("---")
    st.info(
        "💡 **Note:** This page is for managing embeddings of already downloaded ontologies. "
        "To download or update ontologies, use the Update Management page."
    )
    
    # Back button
    if st.button("❌ Back to Home", key="back_from_embed_mgmt"):
        st.session_state.show_ontology_embedding_management = False
        # Clear embedding-related states
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('embedding_progress_', 'confirm_embed_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

# Get enabled ontologies (needed for both main interface and sidebar)
enabled_ontologies = get_enabled_ontologies()

# Only show main interface if no config pages are active
if not any([
    st.session_state.get('show_config_editor', False),
    st.session_state.get('show_health_dashboard', False),
    st.session_state.get('show_embeddings_config', False),
    st.session_state.get('show_ontology_update_management', False),
    st.session_state.get('show_ontology_embedding_management', False)
]):
    # Main resolution interface
    passage = st.text_area("Passage", help="Enter the scientific text you want to map to ontology terms")
    ontology = st.selectbox("Ontology", enabled_ontologies, help="Select the target ontology")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Resolve", type="primary"):
            with st.spinner("Resolving..."):
                resp = requests.post(
                    f"{FASTAPI_URL}/resolve_biocurated_data",
                    json={"passage": passage, "ontology_name": ontology},
                )
            
            if resp.ok:
                data = resp.json()
                if data.get("error"):
                    st.error(f"❌ Error: {data['error']}")
                else:
                    st.success("✅ Resolution Complete")
                    
                    # Display best match
                    best_match = data.get('best_match')
                    if best_match:
                        st.markdown("### 🎯 Best Match")
                        st.markdown(format_ontology_term(best_match))
                        
                        confidence = data.get('confidence')
                        if confidence is not None:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        reason = data.get('reason')
                        if reason:
                            st.markdown(f"**Reasoning:** {reason}")
                        
                        # Display alternatives
                        alternatives = data.get('alternatives', [])
                        if alternatives:
                            st.markdown("### 🔄 Alternative Matches")
                            for i, alt in enumerate(alternatives, 1):
                                with st.expander(f"Alternative {i}: {alt.get('name', 'Unknown')}"):
                                    st.markdown(format_ontology_term(alt))
                    else:
                        st.warning("No matches found")
            else:
                st.error(f"❌ Request failed: {resp.text}")

# Sidebar - Navigation
if st.sidebar.button("🏠 Home", type="primary", use_container_width=True):
    # Clear any session state that might be showing special views
    if 'show_health_dashboard' in st.session_state:
        st.session_state.show_health_dashboard = False
    if 'show_config_editor' in st.session_state:
        st.session_state.show_config_editor = False
    if 'show_embeddings_config' in st.session_state:
        st.session_state.show_embeddings_config = False
    if 'show_ontology_update_management' in st.session_state:
        st.session_state.show_ontology_update_management = False
    if 'show_ontology_embedding_management' in st.session_state:
        st.session_state.show_ontology_embedding_management = False
    if 'config_editor_text' in st.session_state:
        del st.session_state.config_editor_text
    if 'original_config_text' in st.session_state:
        del st.session_state.original_config_text
    if 'embedding_config' in st.session_state:
        del st.session_state.embedding_config
    if 'version_data' in st.session_state:
        del st.session_state.version_data
    if 'update_progress' in st.session_state:
        del st.session_state.update_progress
    if 'editor_version' in st.session_state:
        del st.session_state.editor_version
    # Clear ontology management related states
    keys_to_clear = [key for key in st.session_state.keys() if key.startswith((
        'confirm_', 'embedding_progress_', 'update_progress_', 'download_result_',
        'update_url_', 'confirm_embed_'
    ))]
    for key in keys_to_clear:
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

# Sidebar - Admin Panel
st.sidebar.header("🔧 Admin Panel")
api_key = st.sidebar.text_input(
    "API Key", type="password", value=os.getenv("ADMIN_API_KEY", ""),
    help="Enter your admin API key to access admin features"
)

if api_key:
    # System Health Section
    st.sidebar.subheader("🏥 System Health")
    if st.sidebar.button("Check Health"):
        # Use a placeholder for status messages
        status_placeholder = st.sidebar.empty()
        status_placeholder.info("🔄 Checking system health...")
        
        health_status = check_system_health(api_key)
        
        overall_status = health_status["overall"]
        if overall_status == "healthy":
            status_placeholder.success("🟢 All Systems Healthy")
        elif overall_status == "degraded":
            status_placeholder.warning("🟡 Some Issues Detected")
        else:
            status_placeholder.error("🔴 Critical Issues")
        
        # Store health status in session state for main area display
        st.session_state.health_status = health_status
        st.session_state.show_health_dashboard = True
        # Clear other views
        st.session_state.show_config_editor = False
        if 'config_editor_text' in st.session_state:
            del st.session_state.config_editor_text
        if 'original_config_text' in st.session_state:
            del st.session_state.original_config_text
        st.rerun()
    
    # Ontology Config Section
    st.sidebar.subheader("⚙️ Ontology Config")
    if st.sidebar.button("Edit Configuration"):
        st.session_state.show_config_editor = True
        st.session_state.api_key = api_key  # Store API key for config save
        # Clear other views
        st.session_state.show_health_dashboard = False
        if 'health_status' in st.session_state:
            del st.session_state.health_status
        st.rerun()
    
    # Ontology Management Section
    st.sidebar.subheader("🔄 Ontology Management")
    
    # Update Management button
    if st.sidebar.button("📥 Ontology Updates"):
        st.session_state.show_ontology_update_management = True
        st.session_state.api_key = api_key  # Store API key for operations
        # Clear other views
        st.session_state.show_health_dashboard = False
        st.session_state.show_config_editor = False
        st.session_state.show_embeddings_config = False
        st.session_state.show_ontology_embedding_management = False
        if 'health_status' in st.session_state:
            del st.session_state.health_status
        if 'config_editor_text' in st.session_state:
            del st.session_state.config_editor_text
        if 'original_config_text' in st.session_state:
            del st.session_state.original_config_text
        if 'embedding_config' in st.session_state:
            del st.session_state.embedding_config
        st.rerun()
    
    # Embedding Management button
    if st.sidebar.button("🧠 Manage Embeddings"):
        st.session_state.show_ontology_embedding_management = True
        st.session_state.api_key = api_key  # Store API key for operations
        # Clear other views
        st.session_state.show_health_dashboard = False
        st.session_state.show_config_editor = False
        st.session_state.show_embeddings_config = False
        st.session_state.show_ontology_update_management = False
        if 'health_status' in st.session_state:
            del st.session_state.health_status
        if 'config_editor_text' in st.session_state:
            del st.session_state.config_editor_text
        if 'original_config_text' in st.session_state:
            del st.session_state.original_config_text
        if 'embedding_config' in st.session_state:
            del st.session_state.embedding_config
        st.rerun()
    
    # Embeddings Config Section
    st.sidebar.subheader("🧠 Embeddings Config")
    if st.sidebar.button("Configure Embeddings"):
        st.session_state.show_embeddings_config = True
        st.session_state.api_key = api_key  # Store API key for config save
        # Clear other views
        st.session_state.show_health_dashboard = False
        st.session_state.show_config_editor = False
        st.session_state.show_ontology_update_management = False
        st.session_state.show_ontology_embedding_management = False
        if 'health_status' in st.session_state:
            del st.session_state.health_status
        if 'config_editor_text' in st.session_state:
            del st.session_state.config_editor_text
        if 'original_config_text' in st.session_state:
            del st.session_state.original_config_text
        st.rerun()

# API Testing Interface - only show if no config pages are active
if not any([
    st.session_state.get('show_config_editor', False),
    st.session_state.get('show_health_dashboard', False),
    st.session_state.get('show_embeddings_config', False),
    st.session_state.get('show_ontology_update_management', False),
    st.session_state.get('show_ontology_embedding_management', False)
]):
    st.header("🧪 API Testing Interface")
    st.write("Send mock JSON requests to test the API endpoints")

    # Create tabs for different types of requests
    tab1, tab2, tab3 = st.tabs(["🔍 Resolve Request", "🔄 Admin Update", "📊 Admin Status"])

    with tab1:
        st.subheader("Test Resolution Endpoint")
        
        # Pre-filled examples
        example_requests = {
            "Simple GO query": {
                "passage": "cell division",
                "ontology_name": "GO"
            },
            "Complex biological process": {
                "passage": "regulation of transcription DNA-templated in response to stress",
                "ontology_name": "GO"
            },
            "Disease query": {
                "passage": "inflammatory bowel disease",
                "ontology_name": "DOID"
            }
        }
        
        selected_example = st.selectbox("Load Example Request:", ["Custom"] + list(example_requests.keys()))
        
        if selected_example != "Custom":
            example_data = example_requests[selected_example]
            test_passage = example_data["passage"]
            test_ontology = example_data["ontology_name"]
        else:
            test_passage = ""
            test_ontology = enabled_ontologies[0] if enabled_ontologies else "GO"
        
        # Request editor
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Request Parameters:**")
            test_passage = st.text_area("Passage:", value=test_passage, key="test_passage")
            test_ontology = st.selectbox("Ontology:", enabled_ontologies, 
                                       index=enabled_ontologies.index(test_ontology) if test_ontology in enabled_ontologies else 0,
                                       key="test_ontology")
            
            # Generate request JSON
            request_json = {
                "passage": test_passage,
                "ontology_name": test_ontology
            }
            
            st.markdown("**Generated JSON:**")
            st.code(json.dumps(request_json, indent=2), language="json")
            
            # Store space for confidence progress bar (will be updated after response)
            confidence_placeholder = st.empty()
        
        with col2:
            st.markdown("**Response:**")
            
            if st.button("🚀 Send Test Request", type="primary"):
                if not test_passage.strip():
                    st.error("❌ Passage cannot be empty")
                else:
                    start_time = time.time()
                    
                    with st.spinner("Sending request..."):
                        try:
                            resp = requests.post(
                                f"{FASTAPI_URL}/resolve_biocurated_data",
                                json=request_json,
                                timeout=30
                            )
                            
                            end_time = time.time()
                            response_time = (end_time - start_time) * 1000  # ms
                            
                            # Display response metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Status Code", resp.status_code)
                            with col2:
                                st.metric("Response Time", f"{response_time:.0f} ms")
                            with col3:
                                content_type = resp.headers.get('content-type', 'unknown')
                                st.metric("Content Type", content_type.split(';')[0])
                            
                            # Display response
                            if resp.ok:
                                response_data = resp.json()
                                
                                # Format response nicely
                                if response_data.get("error"):
                                    st.error(f"❌ API Error: {response_data['error']}")
                                else:
                                    st.success("✅ Request Successful")
                                    
                                    # Show key results
                                    best_match = response_data.get('best_match')
                                    if best_match:
                                        st.markdown("**Best Match:**")
                                        st.markdown(format_ontology_term(best_match))
                                        
                                        # Display confidence in the left column placeholder
                                        confidence = response_data.get('confidence')
                                        if confidence is not None:
                                            with confidence_placeholder.container():
                                                st.markdown("**Confidence:**")
                                                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                                        
                                        reason = response_data.get('reason')
                                        if reason:
                                            st.markdown(f"**Reasoning:** {reason}")
                                    
                                    # Alternatives in expandable section
                                    alternatives = response_data.get('alternatives', [])
                                    if alternatives:
                                        with st.expander(f"View {len(alternatives)} Alternative(s)"):
                                            for i, alt in enumerate(alternatives, 1):
                                                st.markdown(f"**{i}.** {format_ontology_term(alt)}")
                                
                                # Raw response in expandable section
                                with st.expander("Raw JSON Response"):
                                    st.json(response_data)
                            
                            else:
                                st.error(f"❌ Request Failed (HTTP {resp.status_code})")
                                st.code(resp.text)
                        
                        except requests.exceptions.Timeout:
                            st.error("❌ Request timed out (30s)")
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Failed to connect to API")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {str(e)}")
    
    with tab2:
        st.subheader("Test Admin Update Endpoint")
        
        if not api_key:
            st.warning("⚠️ Please enter your API key in the sidebar to test admin endpoints")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Update Request Parameters:**")
                update_ontology = st.selectbox("Ontology:", enabled_ontologies, key="admin_test_ont")
                
                # Get default URL
                config = load_ontology_config()
                default_update_url = config.get("ontologies", {}).get(update_ontology, {}).get("default_source_url", "")
                update_url = st.text_input("Source URL:", value=default_update_url, key="admin_test_url")
                
                # Generate request JSON
                update_request_json = {
                    "ontology_name": update_ontology,
                    "source_url": update_url
                }
                
                st.markdown("**Generated JSON:**")
                st.code(json.dumps(update_request_json, indent=2), language="json")
            
            with col2:
                st.markdown("**Response:**")
                
                if st.button("🚀 Send Update Request", type="primary"):
                    if not update_url.strip():
                        st.error("❌ Source URL cannot be empty")
                    else:
                        start_time = time.time()
                        
                        with st.spinner("Sending update request..."):
                            try:
                                resp = requests.post(
                                    f"{FASTAPI_URL}/admin/update_ontology",
                                    headers={"X-API-Key": api_key},
                                    json=update_request_json,
                                    timeout=30
                                )
                                
                                end_time = time.time()
                                response_time = (end_time - start_time) * 1000  # ms
                                
                                # Display response metadata
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Status Code", resp.status_code)
                                with col2:
                                    st.metric("Response Time", f"{response_time:.0f} ms")
                                
                                # Display response
                                if resp.ok:
                                    response_data = resp.json()
                                    st.success("✅ Update Request Successful")
                                    
                                    status_msg = response_data.get("status", "Update initiated")
                                    st.info(f"📝 {status_msg}")
                                    
                                    with st.expander("Raw JSON Response"):
                                        st.json(response_data)
                                else:
                                    st.error(f"❌ Request Failed (HTTP {resp.status_code})")
                                    st.code(resp.text)
                            
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("Test Admin Status Endpoints")
        
        if not api_key:
            st.warning("⚠️ Please enter your API key in the sidebar to test admin endpoints")
        else:
            # Different status endpoints to test
            status_endpoints = {
                "Ontology Status": "/admin/ontology_status",
                "Version Status": "/admin/version_status", 
                "Collections": "/admin/collections",
                "Weaviate Health": "/admin/weaviate_health",
                "OpenAI Health": "/admin/openai_health",
                "Embeddings Config": "/admin/embeddings_config"
            }
            
            selected_endpoint = st.selectbox("Select Status Endpoint:", list(status_endpoints.keys()))
            endpoint_url = status_endpoints[selected_endpoint]
            
            st.markdown(f"**Endpoint:** `GET {endpoint_url}`")
            
            if st.button("🚀 Send Status Request", type="primary"):
                start_time = time.time()
                
                with st.spinner(f"Fetching {selected_endpoint.lower()}..."):
                    try:
                        resp = requests.get(
                            f"{FASTAPI_URL}{endpoint_url}",
                            headers={"X-API-Key": api_key},
                            timeout=15
                        )
                        
                        end_time = time.time()
                        response_time = (end_time - start_time) * 1000  # ms
                        
                        # Display response metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status Code", resp.status_code)
                        with col2:
                            st.metric("Response Time", f"{response_time:.0f} ms")
                        with col3:
                            content_length = len(resp.content)
                            st.metric("Content Size", f"{content_length} bytes")
                        
                        # Display response
                        if resp.ok:
                            try:
                                response_data = resp.json()
                                st.success("✅ Request Successful")
                                
                                # Format specific endpoint responses nicely
                                if selected_endpoint == "Ontology Status":
                                    for ontology, details in response_data.items():
                                        with st.expander(f"📊 {ontology} Status"):
                                            if isinstance(details, dict):
                                                for key, value in details.items():
                                                    st.markdown(f"**{key.replace('_', ' ').title()}:** `{value}`")
                                            else:
                                                st.write(details)
                                
                                elif selected_endpoint == "Collections":
                                    collections = response_data.get('collections', [])
                                    if collections:
                                        for collection in collections:
                                            name = collection.get('name', 'Unknown')
                                            count = collection.get('object_count', 0)
                                            size = collection.get('size_mb', 0)
                                            st.markdown(f"**{name}:** {count:,} objects, {size:.1f} MB")
                                    else:
                                        st.info("No collections found")
                                
                                elif selected_endpoint in ["Weaviate Health", "OpenAI Health"]:
                                    healthy = response_data.get('healthy', False)
                                    details = response_data.get('details', '')
                                    
                                    if healthy:
                                        st.success(f"✅ {selected_endpoint.split()[0]} is healthy: {details}")
                                    else:
                                        st.error(f"❌ {selected_endpoint.split()[0]} has issues: {details}")
                                
                                # Always show raw response
                                with st.expander("Raw JSON Response"):
                                    st.json(response_data)
                            
                            except ValueError:
                                # Not JSON response
                                st.success("✅ Request Successful")
                                st.code(resp.text)
                        
                        else:
                            st.error(f"❌ Request Failed (HTTP {resp.status_code})")
                            st.code(resp.text)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use the API testing interface to validate requests before integrating with external systems.")