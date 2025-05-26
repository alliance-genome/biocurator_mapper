"""Tests for DO embedding generation workflow."""
import pytest
from unittest.mock import Mock, patch

from app.do_embeddings import DOEmbeddingGenerator
from app.openai_client import EmbeddingResult, OpenAIClientError
from app.models import DOTerm
from app.config import Config


class TestDOEmbeddingGenerator:
    """Test DO embedding generation workflow."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock(spec=Config)
        config.get_embeddings_config.return_value = {
            "do_specific": {
                "synonym_types": {
                    "exact_synonym": 1.0,
                    "narrow_synonym": 0.8,
                    "broad_synonym": 0.7,
                    "related_synonym": 0.5
                },
                "include_metadata": {
                    "xref_sources": ["MESH", "ICD10CM", "SNOMEDCT"],
                    "definition_required": True,
                    "include_obsolete": False
                },
                "text_composition": {
                    "primary_text": "name",
                    "context_fields": ["definition", "synonyms"],
                    "separator": " | ",
                    "max_text_length": 8000
                },
                "quality_filters": {
                    "min_definition_length": 10,
                    "exclude_patterns": ["deprecated", "obsolete"]
                }
            },
            "processing": {
                "batch_size": 100,
                "retry_failed": True
            },
            "vectorize_fields": {
                "name": 1.0,
                "definition": 0.8,
                "synonyms": 0.6,
                "xrefs": 0.4
            }
        }
        return config
    
    @pytest.fixture
    def sample_do_term(self):
        """Sample DO term for testing."""
        term = Mock(spec=DOTerm)
        term.id = "DOID:12345"
        term.name = "Test Disease"
        term.definition = "A test disease used for unit testing purposes."
        term.synonyms = {
            "exact_synonym": ["Test Condition"],
            "related_synonym": ["Testing Disease"]
        }
        term.xrefs = ["MESH:D12345", "ICD10CM:A00"]
        term.is_obsolete = False
        return term
    
    @pytest.fixture
    def sample_do_terms(self):
        """Sample list of DO terms."""
        terms = []
        for i in range(3):
            term = Mock(spec=DOTerm)
            term.id = f"DOID:1234{i}"
            term.name = f"Test Disease {i}"
            term.definition = f"A test disease {i} used for unit testing purposes."
            term.synonyms = {"exact_synonym": [f"Test Condition {i}"]}
            term.xrefs = [f"MESH:D1234{i}"]
            term.is_obsolete = False
            terms.append(term)
        return terms
    
    @pytest.fixture
    def mock_embedding_client(self):
        """Mock OpenAI embedding client."""
        client = Mock()
        client.generate_embedding.return_value = EmbeddingResult(
            embedding=[0.1] * 1536,
            text="test text",
            token_count=10,
            model="text-embedding-3-small"
        )
        client.generate_embeddings.return_value = [
            EmbeddingResult(
                embedding=[0.1] * 1536,
                text="test text 1",
                token_count=10,
                model="text-embedding-3-small"
            ),
            EmbeddingResult(
                embedding=[0.2] * 1536,
                text="test text 2",
                token_count=12,
                model="text-embedding-3-small"
            )
        ]
        return client
    
    def test_initialization(self, mock_config):
        """Test DOEmbeddingGenerator initialization."""
        with patch('app.do_embeddings.get_embedding_client') as mock_get_client:
            mock_get_client.return_value = Mock()
            
            generator = DOEmbeddingGenerator(config=mock_config)
            
            assert generator.config == mock_config
            assert 'do_specific' in generator.embeddings_config
            mock_get_client.assert_called_once_with(config=mock_config)
    
    def test_meets_quality_filters_valid_term(self, mock_config, sample_do_term):
        """Test quality filters with valid term."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            assert generator._meets_quality_filters(sample_do_term) is True
    
    def test_meets_quality_filters_short_definition(self, mock_config):
        """Test quality filters with short definition."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Test Disease"
            term.definition = "Short"  # Too short (< 10 chars)
            term.is_obsolete = False
            
            assert generator._meets_quality_filters(term) is False
    
    def test_meets_quality_filters_no_definition_required(self, mock_config):
        """Test quality filters when definition is required but missing."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Test Disease"
            term.definition = None  # Missing definition
            term.is_obsolete = False
            
            assert generator._meets_quality_filters(term) is False
    
    def test_meets_quality_filters_exclude_pattern(self, mock_config):
        """Test quality filters with excluded pattern."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Deprecated Disease"  # Contains excluded pattern
            term.definition = "A deprecated disease for testing."
            term.is_obsolete = False
            
            assert generator._meets_quality_filters(term) is False
    
    def test_meets_quality_filters_obsolete_term(self, mock_config):
        """Test quality filters with obsolete term."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Test Disease"
            term.definition = "A test disease used for unit testing purposes."
            term.is_obsolete = True  # Obsolete term
            
            assert generator._meets_quality_filters(term) is False
    
    def test_preprocess_term(self, mock_config, sample_do_term):
        """Test term preprocessing."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            result = generator._preprocess_term(sample_do_term)
            
            assert "Name: Test Disease" in result
            assert "Definition: A test disease used for unit testing purposes." in result
            assert "Synonyms:" in result
            assert "exact_synonym: Test Condition" in result
            assert " | " in result  # Separator
    
    def test_preprocess_term_with_xrefs(self, mock_config, sample_do_term):
        """Test term preprocessing with cross-references."""
        # Add xrefs to context fields
        config_data = mock_config.get_embeddings_config.return_value
        config_data["do_specific"]["text_composition"]["context_fields"] = ["definition", "synonyms", "xrefs"]
        
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            result = generator._preprocess_term(sample_do_term)
            
            assert "References:" in result
            assert "MESH:D12345" in result
    
    def test_preprocess_term_max_length_truncation(self, mock_config):
        """Test term preprocessing with length truncation."""
        # Set very short max length
        config_data = mock_config.get_embeddings_config.return_value
        config_data["do_specific"]["text_composition"]["max_text_length"] = 50
        
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Test Disease"
            term.definition = "A very long definition that should be truncated because it exceeds the maximum length limit."
            term.synonyms = {}
            
            result = generator._preprocess_term(term)
            
            assert len(result) <= 50
            assert result.endswith("...")
    
    def test_filter_terms(self, mock_config, sample_do_terms):
        """Test term filtering."""
        with patch('app.do_embeddings.get_embedding_client'):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            # Make one term invalid
            sample_do_terms[1].definition = "Short"  # Too short
            
            filtered_terms, rejection_reasons = generator.filter_terms(sample_do_terms)
            
            assert len(filtered_terms) == 2  # 1 rejected
            assert len(rejection_reasons) == 1
            assert "DOID:12341" in rejection_reasons[0]
    
    def test_generate_single_embedding_success(self, mock_config, sample_do_term, mock_embedding_client):
        """Test single embedding generation success."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            result = generator.generate_single_embedding(sample_do_term)
            
            assert result is not None
            assert isinstance(result, EmbeddingResult)
            assert hasattr(result, 'term_id')
            assert result.term_id == "DOID:12345"
            assert hasattr(result, 'term_name')
            assert result.term_name == "Test Disease"
    
    def test_generate_single_embedding_filtered_out(self, mock_config, mock_embedding_client):
        """Test single embedding generation for filtered out term."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            # Create invalid term
            term = Mock(spec=DOTerm)
            term.id = "DOID:12345"
            term.name = "Test Disease"
            term.definition = None  # Missing definition
            term.is_obsolete = False
            
            result = generator.generate_single_embedding(term)
            
            assert result is None
    
    def test_generate_embeddings_batch_success(self, mock_config, sample_do_terms, mock_embedding_client):
        """Test batch embedding generation success."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            results, rejection_reasons = generator.generate_embeddings(sample_do_terms)
            
            assert len(results) == 2  # Only 2 results from mock client
            assert len(rejection_reasons) == 0
            assert all(isinstance(r, EmbeddingResult) for r in results)
            assert all(hasattr(r, 'term_id') for r in results)
    
    def test_generate_embeddings_empty_list(self, mock_config, mock_embedding_client):
        """Test batch embedding generation with empty list."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            results, rejection_reasons = generator.generate_embeddings([])
            
            assert results == []
            assert rejection_reasons == []
    
    def test_generate_embeddings_with_filtering(self, mock_config, sample_do_terms, mock_embedding_client):
        """Test batch embedding generation with filtering."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            # Make one term invalid
            sample_do_terms[1].definition = "Short"  # Too short
            
            results, rejection_reasons = generator.generate_embeddings(sample_do_terms)
            
            assert len(results) == 2  # Still 2 from mock, but 1 term was filtered
            assert len(rejection_reasons) == 1
            assert "DOID:12341" in rejection_reasons[0]
    
    def test_generate_embeddings_without_filtering(self, mock_config, sample_do_terms, mock_embedding_client):
        """Test batch embedding generation without filtering."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            # Make one term invalid
            sample_do_terms[1].definition = "Short"  # Too short
            
            results, rejection_reasons = generator.generate_embeddings(
                sample_do_terms, apply_filters=False
            )
            
            assert len(results) == 2  # Mock returns 2 results
            assert len(rejection_reasons) == 0  # No filtering applied
    
    def test_generate_embeddings_openai_error_no_retry(self, mock_config, sample_do_terms):
        """Test batch embedding generation with OpenAI error when retry is disabled."""
        # Disable retry
        config_data = mock_config.get_embeddings_config.return_value
        config_data["processing"]["retry_failed"] = False
        
        mock_client = Mock()
        mock_client.generate_embeddings.side_effect = OpenAIClientError("API Error")
        
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            with pytest.raises(OpenAIClientError):
                generator.generate_embeddings(sample_do_terms)
    
    def test_generate_embeddings_openai_error_with_retry(self, mock_config, sample_do_terms):
        """Test batch embedding generation with OpenAI error when retry is enabled."""
        mock_client = Mock()
        mock_client.generate_embeddings.side_effect = OpenAIClientError("API Error")
        
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            # With retry enabled, it should continue and return empty results
            results, rejection_reasons = generator.generate_embeddings(sample_do_terms)
            
            assert results == []  # No successful embeddings
            assert len(rejection_reasons) == 0  # Terms passed filtering but API failed
    
    def test_generate_embeddings_custom_batch_size(self, mock_config, sample_do_terms, mock_embedding_client):
        """Test batch embedding generation with custom batch size."""
        with patch('app.do_embeddings.get_embedding_client', return_value=mock_embedding_client):
            generator = DOEmbeddingGenerator(config=mock_config)
            
            results, _ = generator.generate_embeddings(sample_do_terms, batch_size=1)
            
            # Should still work with smaller batches
            assert len(results) <= len(sample_do_terms)
            mock_embedding_client.generate_embeddings.assert_called()