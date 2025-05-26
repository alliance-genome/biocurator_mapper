"""Unit tests for OpenAI API integration components."""

import os
import yaml
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.openai_client import OpenAIEmbeddingClient
from app.do_embeddings import DOEmbeddingGenerator
from app.models import DOTerm  # Assuming this exists

class TestOpenAIConfiguration:
    """Test configuration loading and settings for DO embeddings."""
    
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
            'text-embedding-3-large'
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

class TestOpenAIEmbeddingClient:
    """Test OpenAI Embedding Client functionality."""
    
    @pytest.fixture
    def mock_api_key(self, monkeypatch):
        """Set a mock API key for testing."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test1234")
    
    def test_client_initialization(self, mock_api_key):
        """Test OpenAI client initialization."""
        client = OpenAIEmbeddingClient()
        
        assert client.model == "text-embedding-3-small"
        assert client.client is not None
    
    @patch('openai.Embedding.create')
    def test_generate_embeddings(self, mock_embedding, mock_api_key):
        """Test embedding generation with mocked API response."""
        # Mock embedding response
        mock_embedding.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6])
            ]
        )
        
        client = OpenAIEmbeddingClient()
        texts = ["test text 1", "test text 2"]
        
        embeddings = client.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert len(embeddings[1]) == 3
    
    def test_client_error_handling(self, mock_api_key):
        """Test error handling for API calls."""
        from openai import OpenAIError
        
        with pytest.raises(OpenAIError):
            with patch('openai.Embedding.create', side_effect=OpenAIError("Test error")):
                client = OpenAIEmbeddingClient()
                client.generate_embeddings(["test"])

class TestDOEmbeddingGenerator:
    """Test DO Term Embedding Generator."""
    
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
    
    @patch('app.openai_client.OpenAIEmbeddingClient.generate_embeddings')
    def test_embedding_preprocessing(self, mock_generate_embeddings, sample_do_terms):
        """Test preprocessing of DO terms for embedding generation."""
        # Mock embedding generation
        mock_generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        generator = DOEmbeddingGenerator()
        
        # Manually call _preprocess_term to test it
        preprocessed_texts = [generator._preprocess_term(term) for term in sample_do_terms]
        
        assert len(preprocessed_texts) == 2
        
        # Check that preprocessed text contains expected components
        for text in preprocessed_texts:
            assert "Name:" in text
            assert any(syn_type in text for syn_type in ['Exact', 'Narrow', 'Broad', 'Related'])
    
    def test_embedding_generation_workflow(self, sample_do_terms):
        """Test complete embedding generation workflow."""
        with patch('app.openai_client.OpenAIEmbeddingClient.generate_embeddings') as mock_generate:
            # Mock embedding generation
            mock_generate.return_value = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
            
            generator = DOEmbeddingGenerator()
            embeddings = generator.generate_embeddings(sample_do_terms)
            
            # Verify embeddings generated
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 3
            
            # Verify generate_embeddings was called with preprocessed texts
            mock_generate.assert_called_once()