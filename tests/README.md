# Unit Tests for OpenAI API Integration

## `test_openai_integration.py`

This test suite verifies the implementation of OpenAI API integration for embedding generation, specifically tailored for Disease Ontology (DO) terms.

### Test Coverage

1. **Configuration Tests**
   - Validate embeddings configuration file structure
   - Check DO-specific settings and weights
   - Ensure proper model and processing configurations

2. **OpenAI Embedding Client Tests**
   - Verify client initialization
   - Test embedding generation with mocked API responses
   - Validate error handling mechanisms

3. **DO Embedding Generator Tests**
   - Check term preprocessing workflow
   - Validate complete embedding generation process
   - Ensure configuration-based field weighting works correctly

### Running Tests

```bash
pytest tests/test_openai_integration.py
```

### Mocking and Test Strategies

- Uses `unittest.mock` for API call simulation
- Provides sample DO terms for testing
- Focuses on validating configuration and processing logic