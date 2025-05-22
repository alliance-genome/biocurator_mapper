import os
import asyncio
import requests
import streamlit as st
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

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
    
    with st.expander("Service Details"):
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

# Main resolution interface
passage = st.text_area("Passage", help="Enter the scientific text you want to map to ontology terms")
enabled_ontologies = get_enabled_ontologies()
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
        with st.spinner("Checking system health..."):
            health_status = check_system_health(api_key)
        
        overall_status = health_status["overall"]
        if overall_status == "healthy":
            st.sidebar.success("🟢 All Systems Healthy")
        elif overall_status == "degraded":
            st.sidebar.warning("🟡 Some Issues Detected")
        else:
            st.sidebar.error("🔴 Critical Issues")
        
        # Store health status in session state for main area display
        st.session_state.health_status = health_status
    
    # Display health status in main area if available
    if hasattr(st.session_state, 'health_status'):
        with st.expander("🏥 System Health Dashboard", expanded=False):
            display_health_status(st.session_state.health_status)
    
    # Version Management Section
    st.sidebar.subheader("📊 Version Management")
    if st.sidebar.button("View Versions"):
        with st.spinner("Fetching version information..."):
            try:
                resp = requests.get(
                    f"{FASTAPI_URL}/admin/version_status",
                    headers={"X-API-Key": api_key}
                )
                if resp.ok:
                    version_data = resp.json()
                    st.session_state.version_data = version_data
                    st.sidebar.success("✅ Version data loaded")
                else:
                    st.sidebar.error(f"❌ Failed to fetch versions: {resp.text}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Display version data in main area if available
    if hasattr(st.session_state, 'version_data'):
        with st.expander("📊 Ontology Version Dashboard", expanded=False):
            version_data = st.session_state.version_data
            
            for ontology_name, info in version_data.items():
                st.markdown(f"### {ontology_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Version", info.get('current_version', 'Unknown'))
                with col2:
                    st.metric("Collection", info.get('collection_name', 'None'))
                with col3:
                    last_update = info.get('last_updated', 'Never')
                    if last_update != 'Never':
                        try:
                            # Format datetime if it's a valid timestamp
                            dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            last_update = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass
                    st.metric("Last Updated", last_update)
                
                # Additional details
                if info.get('hash'):
                    st.code(f"Content Hash: {info['hash'][:16]}...")
                
                st.divider()
    
    # Update Ontology Section
    st.sidebar.subheader("🔄 Update Ontology")
    ont_update = st.sidebar.selectbox("Ontology", enabled_ontologies, key="upd")
    
    # Get default source URL for selected ontology
    config = load_ontology_config()
    default_url = config.get("ontologies", {}).get(ont_update, {}).get("default_source_url", "")
    source_url = st.sidebar.text_input("Source URL", value=default_url)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start Update"):
            with st.spinner("Starting update..."):
                r = requests.post(
                    f"{FASTAPI_URL}/admin/update_ontology",
                    headers={"X-API-Key": api_key},
                    json={"ontology_name": ont_update, "source_url": source_url},
                )
            if r.ok:
                st.sidebar.success("✅ Update started")
                response_data = r.json()
                st.sidebar.info(response_data.get("status", "Update initiated"))
            else:
                st.sidebar.error(f"❌ Update failed: {r.text}")
    
    with col2:
        if st.button("Update Progress"):
            with st.spinner("Checking progress..."):
                try:
                    r = requests.get(
                        f"{FASTAPI_URL}/admin/update_progress/{ont_update}",
                        headers={"X-API-Key": api_key}
                    )
                    if r.ok:
                        progress_data = r.json()
                        st.session_state.update_progress = progress_data
                        st.sidebar.success("✅ Progress updated")
                    else:
                        st.sidebar.warning("⚠️ No active update")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Display update progress in main area if available
    if hasattr(st.session_state, 'update_progress'):
        with st.expander("🔄 Update Progress", expanded=True):
            progress_data = st.session_state.update_progress
            
            status = progress_data.get('status', 'unknown')
            if status == 'completed':
                st.success("✅ Update completed successfully")
            elif status == 'failed':
                st.error("❌ Update failed")
            elif status == 'in_progress':
                st.info("⏳ Update in progress...")
            else:
                st.warning(f"⚠️ Status: {status}")
            
            # Progress bar if percentage available
            if 'progress_percentage' in progress_data:
                st.progress(progress_data['progress_percentage'] / 100)
            
            # Recent logs
            logs = progress_data.get('recent_logs', [])
            if logs:
                st.markdown("**Recent Activity:**")
                for log in logs[-10:]:  # Show last 10 logs
                    timestamp = log.get('timestamp', '')
                    message = log.get('message', '')
                    level = log.get('level', 'INFO')
                    
                    if level == 'ERROR':
                        st.error(f"{timestamp}: {message}")
                    elif level == 'WARNING':
                        st.warning(f"{timestamp}: {message}")
                    else:
                        st.info(f"{timestamp}: {message}")
    
    # Collection Management Section
    st.sidebar.subheader("🗂️ Collections")
    if st.sidebar.button("Manage Collections"):
        with st.spinner("Fetching collections..."):
            try:
                resp = requests.get(
                    f"{FASTAPI_URL}/admin/collections",
                    headers={"X-API-Key": api_key}
                )
                if resp.ok:
                    collections_data = resp.json()
                    st.session_state.collections_data = collections_data
                    st.sidebar.success("✅ Collections loaded")
                else:
                    st.sidebar.error(f"❌ Failed to fetch collections: {resp.text}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Display collections in main area if available
    if hasattr(st.session_state, 'collections_data'):
        with st.expander("🗂️ Collection Management", expanded=False):
            collections = st.session_state.collections_data.get('collections', [])
            
            if collections:
                for collection in collections:
                    name = collection.get('name', 'Unknown')
                    count = collection.get('object_count', 0)
                    size = collection.get('size_mb', 0)
                    
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.markdown(f"**{name}**")
                    with col2:
                        st.metric("Objects", f"{count:,}")
                    with col3:
                        st.metric("Size", f"{size:.1f} MB")
                    with col4:
                        if st.button("🗑️", key=f"delete_{name}", help=f"Delete {name}"):
                            if st.session_state.get(f"confirm_delete_{name}"):
                                # Actually delete
                                try:
                                    resp = requests.delete(
                                        f"{FASTAPI_URL}/admin/collections/{name}",
                                        headers={"X-API-Key": api_key}
                                    )
                                    if resp.ok:
                                        st.success(f"✅ Deleted {name}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete: {resp.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                                # Reset confirmation
                                st.session_state[f"confirm_delete_{name}"] = False
                            else:
                                # Ask for confirmation
                                st.session_state[f"confirm_delete_{name}"] = True
                                st.warning(f"Click again to confirm deletion of {name}")
            else:
                st.info("No collections found")
    
    # Ontology Status Section
    st.sidebar.subheader("📋 Current Status")
    if st.sidebar.button("Refresh Status"):
        with st.spinner("Fetching status..."):
            r = requests.get(
                f"{FASTAPI_URL}/admin/ontology_status", headers={"X-API-Key": api_key}
            )
        if r.ok:
            status_data = r.json()
            st.session_state.status_data = status_data
            st.sidebar.success("✅ Status refreshed")
        else:
            st.sidebar.error(f"❌ Failed to fetch status: {r.text}")
    
    # Display formatted status in main area if available
    if hasattr(st.session_state, 'status_data'):
        with st.expander("📋 Detailed Status Information", expanded=False):
            status_data = st.session_state.status_data
            
            # Format the status data nicely
            for ontology, details in status_data.items():
                st.markdown(f"### {ontology}")
                
                if isinstance(details, dict):
                    # Display key metrics
                    if 'collection_name' in details:
                        st.markdown(f"**Collection:** `{details['collection_name']}`")
                    if 'last_updated' in details:
                        st.markdown(f"**Last Updated:** {details['last_updated']}")
                    if 'version' in details:
                        st.markdown(f"**Version:** {details['version']}")
                    if 'object_count' in details:
                        st.markdown(f"**Objects:** {details['object_count']:,}")
                    
                    # Raw data in expandable section
                    with st.expander(f"Raw {ontology} Data"):
                        st.json(details)
                else:
                    st.write(details)
                
                st.divider()

# API Testing Interface
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
                                    
                                    confidence = response_data.get('confidence')
                                    if confidence is not None:
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