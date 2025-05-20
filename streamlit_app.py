import os
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

st.title("Biocurator Mapper")
st.write("Resolve passages to ontology terms using LLM assistance")

passage = st.text_area("Passage")
ontology = st.selectbox("Ontology", ["GO", "DOID"])
if st.button("Resolve"):
    resp = requests.post(
        f"{FASTAPI_URL}/resolve_biocurated_data",
        json={"passage": passage, "ontology_name": ontology},
    )
    if resp.ok:
        st.json(resp.json())
    else:
        st.error(resp.text)

st.sidebar.header("Admin Panel")
api_key = st.sidebar.text_input("API Key", type="password")

st.sidebar.subheader("Update Ontology")
ont_update = st.sidebar.selectbox("Ontology", ["GO", "DOID"], key="upd")
source_url = st.sidebar.text_input("Source URL")
if st.sidebar.button("Trigger Update"):
    r = requests.post(
        f"{FASTAPI_URL}/admin/update_ontology",
        headers={"X-API-Key": api_key},
        json={"ontology_name": ont_update, "source_url": source_url},
    )
    if r.ok:
        st.sidebar.success(r.json())
    else:
        st.sidebar.error(r.text)

st.sidebar.subheader("Ontology Status")
if st.sidebar.button("Refresh Status"):
    r = requests.get(
        f"{FASTAPI_URL}/admin/ontology_status", headers={"X-API-Key": api_key}
    )
    if r.ok:
        st.sidebar.json(r.json())
    else:
        st.sidebar.error(r.text)
