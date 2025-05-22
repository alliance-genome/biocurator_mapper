import os
import asyncio
import requests
import streamlit as st
import yaml

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

st.title("Biocurator Mapper")
st.write("Resolve passages to ontology terms using LLM assistance")

passage = st.text_area("Passage")
enabled_ontologies = get_enabled_ontologies()
ontology = st.selectbox("Ontology", enabled_ontologies)
if st.button("Resolve"):
    with st.spinner("Resolving..."):
        resp = requests.post(
            f"{FASTAPI_URL}/resolve_biocurated_data",
            json={"passage": passage, "ontology_name": ontology},
        )
    if resp.ok:
        data = resp.json()
        if data.get("error"):
            st.error(data["error"])
        else:
            st.success(f"Best match: {data['best_match']['name']} ({data['best_match']['id']})")
            st.write(f"Confidence: {data.get('confidence')}")
            st.write(f"Reason: {data.get('reason')}")
            st.write("Alternatives:")
            for alt in data.get("alternatives", []):
                st.write(f"- {alt['name']} ({alt['id']})")
    else:
        st.error(resp.text)

st.sidebar.header("Admin Panel")
api_key = st.sidebar.text_input(
    "API Key", type="password", value=os.getenv("ADMIN_API_KEY", "")
)

st.sidebar.subheader("Update Ontology")
ont_update = st.sidebar.selectbox("Ontology", enabled_ontologies, key="upd")

# Get default source URL for selected ontology
config = load_ontology_config()
default_url = config.get("ontologies", {}).get(ont_update, {}).get("default_source_url", "")
source_url = st.sidebar.text_input("Source URL", value=default_url)
if st.sidebar.button("Trigger Update"):
    with st.spinner("Starting update..."):
        r = requests.post(
            f"{FASTAPI_URL}/admin/update_ontology",
            headers={"X-API-Key": api_key},
            json={"ontology_name": ont_update, "source_url": source_url},
        )
    if r.ok:
        st.sidebar.success(r.json().get("status"))
    else:
        st.sidebar.error(r.text)

st.sidebar.subheader("Ontology Status")
if st.sidebar.button("Refresh Status"):
    with st.spinner("Fetching status..."):
        r = requests.get(
            f"{FASTAPI_URL}/admin/ontology_status", headers={"X-API-Key": api_key}
        )
    if r.ok:
        st.sidebar.json(r.json())
    else:
        st.sidebar.error(r.text)
