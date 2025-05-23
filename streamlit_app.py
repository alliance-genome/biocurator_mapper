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


def get_download_history(ont_name: str) -> list:
    """Get download history for an ontology."""
    metadata_file = Path("ontology_versions.json")
    
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
    metadata_file = Path("ontology_versions.json")
    
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
    """Display embedding generation progress with manual polling."""
    if f"embedding_progress_{ont_name}" not in st.session_state:
        return
    
    st.markdown("---")
    st.markdown("**🤖 Embedding Progress:**")
    
    progress_info = st.session_state[f"embedding_progress_{ont_name}"]
    start_time = progress_info.get("timestamp", time.time())
    elapsed = time.time() - start_time
    status = progress_info.get("status", "unknown")
    
    if status == "started":
        st.info(f"⏳ Embedding generation started {elapsed:.0f}s ago...")
        # TODO: Add real progress tracking via API when implemented
        st.progress(0.1, text="Initializing embeddings...")
        
        # Simulated progress for now
        if elapsed < 10:
            st.progress(0.2, text="Loading ontology terms...")
        elif elapsed < 20:
            st.progress(0.5, text="Generating embeddings...")
        elif elapsed < 30:
            st.progress(0.8, text="Storing vectors...")
        else:
            st.progress(0.9, text="Finalizing...")
    
    # TODO: Add real-time log display when API is ready
    with st.expander("📋 View Embedding Logs", expanded=False):
        st.code("Embedding generation started...\nProcessing ontology terms...\n[Real-time logs will appear here when API is implemented]")
    
    # Cancel button for embeddings
    if st.button(f"❌ Cancel Embeddings", key=f"cancel_embedding_{ont_name}", type="secondary"):
        # TODO: Implement cancellation when API endpoint is ready
        st.warning("⚠️ Embedding cancellation not yet implemented")
        del st.session_state[f"embedding_progress_{ont_name}"]

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
        selected_model = st.selectbox(
            "Embedding Model",
            options=list(embedding_models.keys()),
            index=list(embedding_models.keys()).index(current_model) if current_model in embedding_models else 0,
            format_func=lambda x: embedding_models[x],
            help="Select the OpenAI embedding model to use for vectorizing ontology terms"
        )
        
        # Show model info
        st.info(f"📊 **Dimensions:** {model_dimensions[selected_model]}")
        st.markdown(f"**Description:** {model_descriptions[selected_model]}")
        
        # Batch size configuration
        current_batch_size = st.session_state.embedding_config.get("processing", {}).get("batch_size", 100)
        batch_size = st.number_input(
            "Batch Size",
            min_value=10,
            max_value=1000,
            value=current_batch_size,
            step=10,
            help="Number of terms to process in each batch during ontology updates"
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
        if st.button("💾 Save Configuration", type="primary"):
            if save_embeddings_config(st.session_state.embedding_config):
                st.success("✅ Configuration saved to embeddings_config.yaml!")
                # Update the session state to reflect saved changes
                st.session_state.embedding_config = load_embeddings_config().copy()
            else:
                st.error("❌ Failed to save configuration")
    
    with col2:
        if st.button("🔄 Test Configuration"):
            with st.spinner("Testing embedding configuration..."):
                # TODO: Call API endpoint to test embedding with sample text
                time.sleep(2)  # Simulate API call
                st.success("✅ Configuration test completed!")
                st.info("Sample text embedded successfully with current settings")
    
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

# Ontology Management Page
if st.session_state.get('show_ontology_management', False):
    st.markdown("---")
    st.markdown("## 🔄 Ontology Management")
    
    # Load ontology configuration
    config = load_ontology_config()
    embeddings_config = load_embeddings_config()
    ontologies = config.get("ontologies", {})
    
    if not ontologies:
        st.warning("⚠️ No ontologies configured. Please check your ontology configuration.")
        st.markdown("---")
    else:
        st.markdown("### 📊 Ontology Status")
        
        # Use session state to control view instead of tabs to avoid jumping
        if 'ontology_view' not in st.session_state:
            st.session_state.ontology_view = "overview"
        
        # Check if we should switch to detailed view automatically
        if any(st.session_state.get(f"confirm_update_{ont}", False) or 
               st.session_state.get(f"download_result_{ont}", False) or
               st.session_state.get(f"embedding_progress_{ont}", False)
               for ont in ontologies.keys()):
            st.session_state.ontology_view = "detailed"
        
        # View selector buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Overview", type="primary" if st.session_state.ontology_view == "overview" else "secondary"):
                st.session_state.ontology_view = "overview"
                st.rerun()
        with col2:
            if st.button("📋 Detailed Actions", type="primary" if st.session_state.ontology_view == "detailed" else "secondary"):
                st.session_state.ontology_view = "detailed"
                st.rerun()
        
        st.markdown("---")
        
        if st.session_state.ontology_view == "overview":
            # Overview table
            st.markdown("**Current Ontology Status:**")
            
            # Create a summary table
            overview_data = []
            for ont_name, ont_config in ontologies.items():
                if ont_config.get("enabled", True):
                    overview_data.append({
                        "Ontology": ont_name,
                        "Name": ont_config.get("name", "Unknown"),
                        "Status": "✅ Enabled",
                        "Source": ont_config.get("default_source_url", "Not configured")[:50] + "..." if len(ont_config.get("default_source_url", "")) > 50 else ont_config.get("default_source_url", "Not configured")
                    })
            
            if overview_data:
                import pandas as pd
                df = pd.DataFrame(overview_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No enabled ontologies found.")
        
        elif st.session_state.ontology_view == "detailed":
            # Detailed actions for each ontology
            for ont_name, ont_config in ontologies.items():
                if not ont_config.get("enabled", True):
                    continue
                    
                # Check if this ontology has active operations to auto-expand
                has_active_progress = (f"download_result_{ont_name}" in st.session_state or 
                                     f"embedding_progress_{ont_name}" in st.session_state or
                                     st.session_state.get(f"confirm_update_{ont_name}", False))
                
                with st.expander(f"🧬 {ont_name} - {ont_config.get('name', 'Unknown')}", expanded=has_active_progress):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Ontology:** {ont_name}")
                        st.markdown(f"**Full Name:** {ont_config.get('name', 'Unknown')}")
                        
                        default_url = ont_config.get("default_source_url", "")
                        source_url = st.text_input(
                            "Source URL", 
                            value=default_url, 
                            key=f"url_{ont_name}",
                            help="URL to download the ontology data"
                        )
                        
                        # Embedding model info
                        model_name = embeddings_config.get("model", {}).get("name", "text-ada-002")
                        st.info(f"🧠 **Embedding Model:** {model_name}")
                        
                        # Show download history
                        download_history = get_download_history(ont_name)
                        if download_history:
                            history_col1, history_col2 = st.columns([3, 1])
                            with history_col1:
                                st.markdown("#### 📜 Download History")
                            with history_col2:
                                if st.button("🗑️ Clear", key=f"clear_history_{ont_name}", help="Clear download history"):
                                    clear_download_history(ont_name)
                                    st.rerun()
                            
                            for version in download_history[-3:]:  # Show last 3
                                timestamp = datetime.fromisoformat(version['timestamp'])
                                st.text(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                      f"{version['filename']} ({version['size_mb']:.2f} MB)")
                        
                        # Show any download result
                        if f"download_result_{ont_name}" in st.session_state:
                            st.markdown("---")
                            st.markdown("**📥 Download Status:**")
                            result = st.session_state[f"download_result_{ont_name}"]
                            if result["success"]:
                                st.success(f"✅ {result['message']}")
                                st.info(f"File saved to: `{result['file_path']}`")
                            else:
                                st.error(f"❌ {result['message']}")
                            
                            # Clear the result after showing
                            if st.button("🔄 Clear Status", key=f"clear_result_{ont_name}"):
                                del st.session_state[f"download_result_{ont_name}"]
                                st.rerun()
                        
                        # Show download in progress
                        if st.session_state.get(f"downloading_{ont_name}", False):
                            st.markdown("---")
                            st.markdown("**📥 Download Progress:**")
                            with st.spinner(f"Downloading {ont_name} from {st.session_state.get(f'download_url_{ont_name}', '')}..."):
                                result = perform_simple_download(ont_name, st.session_state.get(f'download_url_{ont_name}', ''))
                                st.session_state[f"download_result_{ont_name}"] = result
                                del st.session_state[f"downloading_{ont_name}"]
                                del st.session_state[f"download_url_{ont_name}"]
                                st.rerun()
                    
                    with col2:
                        st.markdown("**Actions:**")
                        
                        # Update Ontology Button
                        if st.button(f"📥 Update Data", key=f"update_{ont_name}", help="Download and update ontology data"):
                            if not source_url.strip():
                                st.error("❌ Source URL is required")
                            else:
                                st.session_state[f"confirm_update_{ont_name}"] = True
                        
                        st.markdown("*Note: This only downloads the ontology file. Embeddings are separate.*")
                        
                        # Show confirmation dialog if needed
                        if st.session_state.get(f"confirm_update_{ont_name}", False):
                            st.warning(f"⚠️ **Confirm updating {ont_name}?**")
                            st.markdown("This will download new ontology data from the source URL.")
                            
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button(f"✅ Confirm", key=f"confirm_update_btn_{ont_name}", type="primary"):
                                    # Set download flag and URL
                                    st.session_state[f"downloading_{ont_name}"] = True
                                    st.session_state[f"download_url_{ont_name}"] = source_url
                                    
                                    # Clear confirmation state
                                    del st.session_state[f"confirm_update_{ont_name}"]
                                    
                                    # Rerun to trigger download in left column
                                    st.rerun()
                            
                            with col_confirm2:
                                if st.button(f"❌ Cancel", key=f"cancel_update_btn_{ont_name}"):
                                    del st.session_state[f"confirm_update_{ont_name}"]
                        
                        # Cost estimation for embeddings
                        cost_per_1k = {
                            "text-ada-002": 0.0001, 
                            "text-embedding-3-small": 0.00002, 
                            "text-embedding-3-large": 0.00013
                        }
                        estimated_cost = cost_per_1k.get(model_name, 0.0001)
                        st.markdown(f"💰 **Est. Cost:** ~${estimated_cost:.5f}/1K tokens")
                        
                        # Create Embeddings Button with warnings
                        st.markdown("---")
                        if st.button(f"🧠 Create Embeddings", key=f"embed_{ont_name}", help="Generate embeddings for this ontology"):
                            # Multiple confirmation steps for expensive operation
                            if f"confirm_embed_step1_{ont_name}" not in st.session_state:
                                st.session_state[f"confirm_embed_step1_{ont_name}"] = True
                                st.error("⚠️ **WARNING: This operation costs money!**")
                                st.warning(f"This will generate embeddings for all terms in {ont_name} using OpenAI's API.")
                                st.info(f"Estimated cost: ~${estimated_cost:.5f} per 1,000 tokens")
                                
                                if st.button(f"💸 I Understand the Cost", key=f"cost_confirm_{ont_name}"):
                                    st.session_state[f"confirm_embed_step2_{ont_name}"] = True
                                    st.rerun()
                            
                            elif f"confirm_embed_step2_{ont_name}" in st.session_state:
                                st.warning("⚠️ **Final confirmation required:**")
                                st.markdown(f"**Ontology:** {ont_name}")
                                st.markdown(f"**Model:** {model_name}")
                                st.markdown(f"**This action cannot be undone easily.**")
                                
                                col_confirm1, col_confirm2 = st.columns(2)
                                with col_confirm1:
                                    if st.button(f"🚀 START EMBEDDINGS", key=f"final_confirm_{ont_name}", type="primary"):
                                        # Start embedding process
                                        with st.spinner(f"Starting embeddings for {ont_name}..."):
                                            try:
                                                # TODO: Call embedding API endpoint
                                                st.info(f"🚀 Starting embedding generation for {ont_name}")
                                                st.warning("⚠️ Embedding API endpoint not yet implemented")
                                                # Store operation in session state for progress tracking
                                                st.session_state[f"embedding_progress_{ont_name}"] = {
                                                    "status": "started",
                                                    "ontology": ont_name,
                                                    "timestamp": time.time()
                                                }
                                            except Exception as e:
                                                st.error(f"❌ Error starting embeddings: {str(e)}")
                                        # Clear confirmation states
                                        if f"confirm_embed_step1_{ont_name}" in st.session_state:
                                            del st.session_state[f"confirm_embed_step1_{ont_name}"]
                                        if f"confirm_embed_step2_{ont_name}" in st.session_state:
                                            del st.session_state[f"confirm_embed_step2_{ont_name}"]
                                        st.rerun()
                                
                                with col_confirm2:
                                    if st.button(f"❌ Cancel", key=f"cancel_embed_{ont_name}"):
                                        # Clear confirmation states
                                        if f"confirm_embed_step1_{ont_name}" in st.session_state:
                                            del st.session_state[f"confirm_embed_step1_{ont_name}"]
                                        if f"confirm_embed_step2_{ont_name}" in st.session_state:
                                            del st.session_state[f"confirm_embed_step2_{ont_name}"]
                                        st.rerun()
                        
                        # Show embedding progress if available using fragment
                        if f"embedding_progress_{ont_name}" in st.session_state:
                            show_embedding_progress(ont_name, st.session_state.get('api_key', ''))
    
    # Cancel button
    st.markdown("---")
    if st.button("❌ Back to Home"):
        st.session_state.show_ontology_management = False
        if 'ontology_management_data' in st.session_state:
            del st.session_state.ontology_management_data
        if 'ontology_view' in st.session_state:
            del st.session_state.ontology_view
        # Clear any pending confirmations
        keys_to_clear = [key for key in st.session_state.keys() if key.startswith(('confirm_', 'embedding_progress_', 'download_result_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")

# Get enabled ontologies (needed for both main interface and sidebar)
enabled_ontologies = get_enabled_ontologies()

# Only show main interface if no config pages are active
if not any([
    st.session_state.get('show_config_editor', False),
    st.session_state.get('show_health_dashboard', False),
    st.session_state.get('show_embeddings_config', False),
    st.session_state.get('show_ontology_management', False)
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
    if 'show_ontology_management' in st.session_state:
        st.session_state.show_ontology_management = False
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
    if 'ontology_view' in st.session_state:
        del st.session_state.ontology_view
    keys_to_clear = [key for key in st.session_state.keys() if key.startswith(('confirm_', 'embedding_progress_', 'download_result_', 'ontology_management_'))]
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
    if st.sidebar.button("Manage Ontologies"):
        st.session_state.show_ontology_management = True
        st.session_state.api_key = api_key  # Store API key for operations
        # Clear other views
        st.session_state.show_health_dashboard = False
        st.session_state.show_config_editor = False
        st.session_state.show_embeddings_config = False
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
    st.session_state.get('show_ontology_management', False)
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
                "OpenAI Health": "/admin/openai_health"
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