# DO Term Extraction End-to-End Tests

This document describes the end-to-end tests for verifying DO (Disease Ontology) term extraction functionality.

## Test File: `test_do_term_extraction_e2e.py`

### Purpose
These tests verify that the fix for DO term synonym extraction is working correctly in the full application context. They ensure that DO terms are properly extracted with all synonym types (exact, narrow, broad, related) and that this data flows correctly through the UI.

### Test Cases

1. **`test_do_download_extracts_synonyms`**
   - Downloads DO ontology through the UI
   - Verifies that terms are extracted with proper counts (>10,000 terms expected)
   - Confirms that the extraction includes synonym data

2. **`test_do_embeddings_include_synonyms`**
   - Tests the embeddings generation workflow
   - Verifies that synonym data is included in the embedding process
   - Monitors status messages during generation

3. **`test_verify_searchable_text_generation`**
   - Confirms that searchable text is properly built
   - Verifies that terms are ready for embedding with rich content

4. **`test_embeddings_config_shows_synonym_options`**
   - Checks the embeddings configuration UI
   - Verifies field vectorization options are available

### Prerequisites

1. **Application must be running:**
   ```bash
   docker compose up
   ```

2. **Admin API Key Configuration:**
   The tests require an admin API key to access protected endpoints. This can be provided in two ways:
   
   - **Local Development**: Create a `.env` file with `ADMIN_API_KEY=your-key`
   - **CI/CD**: Set the `ADMIN_API_KEY` environment variable
   
   The tests will automatically load from `.env` file if present, or use the environment variable.

3. **Other Environment variables (optional):**
   ```bash
   export STREAMLIT_URL=http://localhost:8501  # Default
   export FASTAPI_URL=http://localhost:8000     # Default
   ```

### Running the Tests

Run all DO extraction tests:
```bash
pytest tests/e2e/test_do_term_extraction_e2e.py -v
```

Run a specific test:
```bash
pytest tests/e2e/test_do_term_extraction_e2e.py::TestDOTermExtractionE2E::test_do_download_extracts_synonyms -v
```

Run with browser visible (non-headless):
```bash
pytest tests/e2e/test_do_term_extraction_e2e.py -v --headed
```

Run with detailed output:
```bash
pytest tests/e2e/test_do_term_extraction_e2e.py -v -s
```

### What These Tests Verify

1. **Data Extraction**: DO terms are extracted with all synonym types
2. **UI Integration**: The fixed extraction method works correctly in the UI
3. **Embeddings Flow**: Synonym data flows through to embeddings generation
4. **Configuration**: The UI properly supports synonym configuration

### Troubleshooting

- **Connection Refused**: Ensure the application is running (`docker compose up`)
- **Authentication Failed**: Check ADMIN_API_KEY environment variable
- **Timeout Errors**: Increase timeout values in the test or check network speed
- **Screenshot Debugging**: Tests save screenshots to help debug issues

### Related Files

- `/app/ontology_manager.py`: Contains the fixed `_extract_enhanced_term_data` method
- `/app/go_parser.py`: Parser that handles DO term extraction
- `/tests/test_ontology_manager_do_extraction.py`: Unit tests for the extraction
- `/tests/test_doid_parsing.py`: Tests for DO parsing compatibility