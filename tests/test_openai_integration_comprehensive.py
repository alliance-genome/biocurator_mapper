"""Comprehensive unit tests for OpenAI API integration components."""

import os
import yaml
import pytest
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import httpx

from app.openai_client import OpenAIEmbeddingClient, get_embedding_client
from app.do_embeddings import DOEmbeddingGenerator
from app.models import DOTerm


class TestConfigurationSettings:
    """Test configuration loading and validation for DO-specific embeddings settings."""
    
    def test_config_file_loading(self):
        """Verify embeddings configuration can be loaded correctly."""
        config_path = Path(__file__).parent.parent / 'embeddings_config.yaml'
        
        # Verify file exists
        assert config_path.exists(), f"Configuration file not found at {config_path}"
        
        # Load and validate configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key configuration sections
        assert 'model' in config, "Model configuration missing"
        assert 'processing' in config, "Processing configuration missing"
        assert 'vectorize_fields' in config, "Vectorize fields configuration missing"
        
        # Validate specific settings
        assert config['model']['name'] in [
            'text-embedding-3-small', 
            'text-embedding-3-large',
            'text-ada-002'
        ], "Invalid embedding model selected"
        
        assert config['processing'].get('batch_size', 0) > 0, "Invalid batch size"
    
    def test_do_specific_configuration(self):
        """Verify DO-specific configuration settings."""
        config_path = Path(__file__).parent.parent / 'embeddings_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check DO-specific configuration
        do_specific = config.get('do_specific', {})
        
        assert 'synonym_types' in do_specific, "Synonym type weights missing"
        
        synonym_types = do_specific['synonym_types']
        expected_types = ['exact_synonym', 'narrow_synonym', 'broad_synonym', 'related_synonym']
        
        for syn_type in expected_types:
            assert syn_type in synonym_types, f"{syn_type} missing from synonym types"
            assert 0 <= synonym_types[syn_type] <= 1, f"Invalid weight for {syn_type}"
    
    def test_vectorize_fields_configuration(self):
        """Test vectorize fields configuration and weights."""
        config_path = Path(__file__).parent.parent / 'embeddings_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vectorize_fields = config.get('vectorize_fields', {})
        
        # Check required fields exist
        required_fields = ['name', 'definition', 'synonyms']
        for field in required_fields:
            assert field in vectorize_fields, f"Required field {field} missing"
        
        # Check weights are valid
        for field, weight in vectorize_fields.items():
            if isinstance(weight, (int, float)):
                assert 0 <= weight <= 1, f"Invalid weight for {field}: {weight}"
    
    def test_missing_config_file_handling(self):
        """Test behavior when configuration file is missing."""
        generator = DOEmbeddingGenerator(config_path="/nonexistent/path/config.yaml")
        
        # Should still initialize with empty config
        assert isinstance(generator.config, dict)
    
    def test_invalid_yaml_handling(self):
        """Test behavior when YAML file is malformed."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: ][")
            invalid_config_path = f.name
        
        try:
            generator = DOEmbeddingGenerator(config_path=invalid_config_path)
            # Should handle YAML error gracefully
            assert isinstance(generator.config, dict)
        finally:
            os.unlink(invalid_config_path)


class TestOpenAIAPIClient:
    """Test OpenAI API client initialization and basic functionality."""
    
    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        """Set a mock API key for testing."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234567890")
    
    def test_client_initialization_with_api_key(self, mock_api_key):
        """Test OpenAI client initialization with API key."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            client = OpenAIEmbeddingClient()
            
            assert client.model == "text-embedding-3-small"
            assert client.max_retries == 3
            assert client.base_wait_seconds == 1.0
            assert client.max_wait_seconds == 60.0
    
    def test_client_initialization_without_api_key(self, monkeypatch):
        """Test that client raises error without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="OpenAI API key must be provided"):
            OpenAIEmbeddingClient()
    
    def test_client_initialization_with_custom_params(self, mock_api_key):
        """Test client initialization with custom parameters."""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            client = OpenAIEmbeddingClient(
                model="text-embedding-3-large",
                max_retries=5,
                base_wait_seconds=2.0,
                max_wait_seconds=120.0
            )
            
            assert client.model == "text-embedding-3-large"
            assert client.max_retries == 5
            assert client.base_wait_seconds == 2.0
            assert client.max_wait_seconds == 120.0
    
    @patch('app.openai_client.OpenAI')
    def test_connection_validation(self, mock_openai, mock_api_key):
        """Test connection validation during initialization."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock successful validation
        mock_client.embeddings.create.return_value = MagicMock()
        
        client = OpenAIEmbeddingClient()
        
        # Verify validation was called
        mock_client.embeddings.create.assert_called_once()
    
    @patch('app.openai_client.OpenAI')
    def test_generate_embeddings_success(self, mock_openai, mock_api_key):
        """Test successful embedding generation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock embedding response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        client = OpenAIEmbeddingClient()
        texts = ["test text 1", "test text 2"]
        
        embeddings = client.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert len(embeddings[1]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    def test_get_embedding_client_function(self, mock_api_key):
        """Test the convenience function for getting an embedding client."""
        with patch('app.openai_client.OpenAIEmbeddingClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            client = get_embedding_client(model="text-embedding-3-large")
            
            mock_client_class.assert_called_once_with(model="text-embedding-3-large")
            assert client == mock_client


class TestEmbeddingGenerationWorkflow:
    """Test DO term embedding generation workflow and preprocessing."""
    
    @pytest.fixture
    def sample_do_terms(self):
        """Create sample DO terms for testing."""
        return [
            DOTerm(
                id="DOID:1234", 
                name="Sample Disease", 
                definition="A sample disease for testing",
                synonyms={
                    'exact_synonym': ['Exact Disease'],
                    'narrow_synonym': ['Specific Disease'],
                    'broad_synonym': ['General Condition']
                }
            ),
            DOTerm(
                id="DOID:5678", 
                name="Another Disease", 
                definition="Another sample disease for testing",
                synonyms={
                    'related_synonym': ['Related Condition']
                }
            )
        ]
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'vectorize_fields': {
                'name': 1.0,
                'definition': 0.8,
                'synonyms': 0.6
            },
            'do_specific': {
                'synonym_types': {
                    'exact_synonym': 1.0,
                    'narrow_synonym': 0.8,
                    'broad_synonym': 0.7,
                    'related_synonym': 0.5
                }
            },
            'preprocessing': {
                'combine_fields_separator': ' | '
            },
            'processing': {
                'batch_size': 2
            },
            'model': {
                'dimensions': 1536
            }
        }
    
    @patch('app.do_embeddings.get_embedding_client')
    def test_generator_initialization(self, mock_get_client, mock_config):
        """Test DO embedding generator initialization."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        with patch.object(DOEmbeddingGenerator, '_load_config', return_value=mock_config):
            generator = DOEmbeddingGenerator()
            
            assert generator.config == mock_config
            assert generator.dimensions == 1536
    
    @patch('app.do_embeddings.get_embedding_client')
    def test_term_preprocessing(self, mock_get_client, sample_do_terms, mock_config):
        """Test preprocessing of DO terms for embedding generation."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        with patch.object(DOEmbeddingGenerator, '_load_config', return_value=mock_config):
            generator = DOEmbeddingGenerator()
            
            # Test preprocessing of first term
            preprocessed = generator._preprocess_term(sample_do_terms[0])
            
            assert "Name: Sample Disease" in preprocessed
            assert "Definition: A sample disease for testing" in preprocessed
            assert "Exact Synonym: Exact Disease" in preprocessed
            assert "Narrow Synonym: Specific Disease" in preprocessed
            assert "Broad Synonym: General Condition" in preprocessed
            assert " | " in preprocessed  # Check separator
    
    @patch('app.do_embeddings.get_embedding_client')
    def test_embedding_generation_workflow(self, mock_get_client, sample_do_terms, mock_config):
        """Test complete embedding generation workflow."""
        mock_client = MagicMock()
        mock_client.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_get_client.return_value = mock_client
        
        with patch.object(DOEmbeddingGenerator, '_load_config', return_value=mock_config):
            generator = DOEmbeddingGenerator()
            
            embeddings = generator.generate_embeddings(sample_do_terms)
            
            # Verify embeddings generated
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 3
            
            # Verify generate_embeddings was called with correct parameters
            mock_client.generate_embeddings.assert_called_once()
            call_args = mock_client.generate_embeddings.call_args
            assert len(call_args[1]['texts']) == 2  # Two preprocessed texts
            assert call_args[1]['dimensions'] == 1536
    
    @patch('app.do_embeddings.get_embedding_client')
    def test_batch_processing(self, mock_get_client, mock_config):
        """Test batch processing of embeddings."""
        mock_client = MagicMock()
        mock_client.generate_embeddings.side_effect = [
            [[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]
        ]
        mock_get_client.return_value = mock_client
        
        # Create 5 terms to test batching with batch_size=2
        terms = [
            DOTerm(id=f"DOID:{i}", name=f"Disease {i}", definition=f"Definition {i}")
            for i in range(5)
        ]
        
        with patch.object(DOEmbeddingGenerator, '_load_config', return_value=mock_config):
            generator = DOEmbeddingGenerator()
            
            embeddings = generator.generate_embeddings(terms, batch_size=2)
            
            # Should have 3 batches: [2, 2, 1]
            assert mock_client.generate_embeddings.call_count == 3
            assert len(embeddings) == 3  # Total embeddings returned


class TestErrorHandlingAndRateLimiting:
    """Test API error handling, retry logic, and rate limiting mechanisms."""
    
    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        """Set a mock API key for testing."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234567890")
    
    @patch('app.openai_client.OpenAI')
    def test_rate_limit_error_handling(self, mock_openai, mock_api_key):
        """Test handling of rate limit errors."""
        from openai import OpenAIError
        
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create a rate limit error
        rate_limit_error = OpenAIError("Rate limit exceeded")
        rate_limit_error.http_status = 429
        
        mock_client.embeddings.create.side_effect = rate_limit_error
        
        client = OpenAIEmbeddingClient()
        
        with pytest.raises(OpenAIError):
            client.generate_embeddings(["test"])
    
    @patch('app.openai_client.OpenAI')
    def test_network_error_handling(self, mock_openai, mock_api_key):
        """Test handling of network errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create a network error
        network_error = httpx.RequestError("Network error")
        mock_client.embeddings.create.side_effect = network_error
        
        client = OpenAIEmbeddingClient()
        
        with pytest.raises(httpx.RequestError):
            client.generate_embeddings(["test"])
    
    @patch('app.openai_client.OpenAI')
    def test_timeout_error_handling(self, mock_openai, mock_api_key):
        """Test handling of timeout errors."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create a timeout error
        timeout_error = TimeoutError("Request timed out")
        mock_client.embeddings.create.side_effect = timeout_error
        
        client = OpenAIEmbeddingClient()
        
        with pytest.raises(TimeoutError):
            client.generate_embeddings(["test"])
    
    @patch('app.openai_client.OpenAI')
    def test_retry_configuration(self, mock_openai, mock_api_key):
        """Test retry configuration is properly set."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = OpenAIEmbeddingClient(max_retries=5)
        
        # Verify OpenAI client was initialized with correct max_retries
        mock_openai.assert_called_with(
            api_key="sk-test1234567890",
            max_retries=5,
            timeout=httpx.Timeout(60.0, read=30.0, write=30.0, connect=5.0)
        )
    
    @patch('app.openai_client.OpenAI')
    def test_custom_timeout_configuration(self, mock_openai, mock_api_key):
        """Test custom timeout configuration."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        custom_timeout = httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0)
        client = OpenAIEmbeddingClient()
        
        # Verify timeout was set during initialization
        call_args = mock_openai.call_args[1]
        assert 'timeout' in call_args
        assert isinstance(call_args['timeout'], httpx.Timeout)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])