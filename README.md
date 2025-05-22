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

## Ontology Configuration

The system uses two types of configuration:

### Static Configuration (`ontology_config.yaml`)
Defines ontology schemas and parsing rules:
- Ontology definitions (GO, DOID, etc.)
- Default source URLs for ontology downloads
- ID format parsing rules
- JSON parsing configuration

You can modify this file to add new ontologies or change existing settings without code changes.

### Runtime State (`ontology_versions.json`)
Tracks currently loaded ontology versions:
- Which Weaviate collections are active
- Last update timestamps
- Source URLs for loaded data

This file is automatically created and managed by the application.

## Testing

### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-asyncio pytest-mock pytest-cov httpx
```

### Running Tests

Run all tests:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

Run specific test suites:
```bash
make test-config              # Configuration tests
make test-models              # Pydantic model tests  
make test-go-parsing          # GO data parsing tests
make test-yaml-integration    # YAML config integration tests
```

Or use pytest directly:
```bash
pytest tests/test_config.py -v
pytest tests/test_models.py -v
pytest tests/test_go_data_parsing.py -v
pytest tests/test_yaml_config_integration.py -v
```

Common tasks:
```bash
make test          # Run all tests (34 tests)
make test-cov      # Run with HTML coverage report
make test-fast     # Run only fast tests
make clean         # Clean up generated files
```

### Basic Test Runner

If you don't have pytest installed, you can run basic validation tests:
```bash
python3 run_tests.py
```

### Continuous Integration

This project uses GitHub Actions for automated testing:

- **GitHub Actions**: `.github/workflows/test.yml` - Runs on Python 3.8-3.11
- Tests automatically run on pull requests and pushes to the `main` branch
- Includes coverage reporting with Codecov integration

## Ontology Embeddings

Use the admin endpoint `/admin/update_ontology` to load ontologies. The system now supports configurable ontology sources via the YAML configuration file.
