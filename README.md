# Biocurator Mapper

This project provides a simple service for resolving biological text passages to ontology terms. It combines a Weaviate vector database with OpenAI language models and exposes functionality via a FastAPI API and a basic Streamlit user interface.

## Environment Variables

- `OPENAI_API_KEY` – API key for OpenAI.
- `ADMIN_API_KEY` – key required for admin endpoints.
- `WEAVIATE_URL` – URL for the Weaviate instance (default `http://weaviate:8080`).

## Running with Docker

Build and start the services using docker compose:

```bash
docker compose build
OPENAI_API_KEY=sk-... ADMIN_API_KEY=secret docker compose up
```

The FastAPI service will be available at `http://localhost:8000` and the Streamlit UI at `http://localhost:8501`.
Configuration data such as `ontology_versions.json` is stored in a Docker volume (`app_config_data`), so the file may not appear on your host filesystem.

## API Usage

Example request using `curl`:

```bash
curl -X POST http://localhost:8000/resolve_biocurated_data \
    -H 'Content-Type: application/json' \
    -d '{"passage": "apoptosis of T cells", "ontology_name": "GO"}'
```

Admin endpoints require the `X-API-Key` header with the value of `ADMIN_API_KEY`.

## Streamlit UI

Run `docker compose up streamlit` (or the whole stack). Open `http://localhost:8501` in your browser to interact with the API and manage ontology updates.

## Ontology Embeddings

Use the admin endpoint `/admin/update_ontology` to load the small test Gene Ontology included with this repository. For real ontologies you would run an external ingestion pipeline to parse the data and populate Weaviate.
