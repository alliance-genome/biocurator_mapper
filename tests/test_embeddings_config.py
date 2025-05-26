"""Tests for embeddings configuration loading and validation."""
import pytest
import yaml
from unittest.mock import patch, mock_open

from app.config import Config


class TestEmbeddingsConfig:
    """Test embeddings configuration loading and validation."""
    
    def test_load_embeddings_config(self):
        """Test that embeddings config loads successfully."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        assert embeddings_config is not None
        assert "model" in embeddings_config
        assert "do_specific" in embeddings_config
        
    def test_model_configuration(self):
        """Test model configuration validation."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        model_config = embeddings_config["model"]
        assert model_config["name"] == "text-embedding-3-small"
        assert model_config["dimensions"] == 1536
        
    def test_do_specific_synonym_types(self):
        """Test DO-specific synonym type weights."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        synonym_types = embeddings_config["do_specific"]["synonym_types"]
        assert synonym_types["exact_synonym"] == 1.0
        assert synonym_types["narrow_synonym"] == 0.8
        assert synonym_types["broad_synonym"] == 0.7
        assert synonym_types["related_synonym"] == 0.5
        
    def test_do_specific_metadata_config(self):
        """Test DO-specific metadata configuration."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        metadata_config = embeddings_config["do_specific"]["include_metadata"]
        assert metadata_config["definition_required"] is True
        assert metadata_config["include_obsolete"] is False
        assert "MESH" in metadata_config["xref_sources"]
        assert "ICD10CM" in metadata_config["xref_sources"]
        
    def test_text_composition_config(self):
        """Test text composition configuration."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        text_config = embeddings_config["do_specific"]["text_composition"]
        assert text_config["primary_text"] == "name"
        assert "definition" in text_config["context_fields"]
        assert "synonyms" in text_config["context_fields"]
        assert text_config["max_text_length"] == 8000
        
    def test_quality_filters_config(self):
        """Test quality filters configuration."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        filters = embeddings_config["do_specific"]["quality_filters"]
        assert filters["min_definition_length"] == 10
        assert "deprecated" in filters["exclude_patterns"]
        assert "obsolete" in filters["exclude_patterns"]
        
    def test_processing_config(self):
        """Test processing configuration validation."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        processing = embeddings_config["processing"]
        assert processing["batch_size"] == 100
        assert processing["parallel_processing"] is True
        assert processing["max_retries"] == 3
        
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(yaml.YAMLError):
                config = Config()
                config.get_embeddings_config()
                
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with patch("builtins.open", side_effect=FileNotFoundError("Config file not found")):
            config = Config()
            # Should return default config instead of raising error
            embeddings_config = config.get_embeddings_config()
            assert embeddings_config is not None
            assert "model" in embeddings_config
                
    def test_vectorize_fields_weights(self):
        """Test vectorize fields weight configuration."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        fields = embeddings_config["vectorize_fields"]
        assert fields["name"] == 1.0
        assert fields["definition"] == 0.8
        assert fields["synonyms"] == 0.6
        assert fields["xrefs"] == 0.4
        
    def test_performance_settings(self):
        """Test performance settings configuration."""
        config = Config()
        embeddings_config = config.get_embeddings_config()
        
        performance = embeddings_config["performance"]
        assert performance["request_timeout"] == 30
        assert performance["rate_limit_delay"] == 0.1