"""Tests for OpenAI API client."""
import pytest
import os
from unittest.mock import Mock, patch

from app.openai_client import (
    OpenAIEmbeddingClient, 
    OpenAIClientError, 
    EmbeddingResult,
    get_embedding_client,
    OPENAI_AVAILABLE
)
from app.config import Config


class TestOpenAIEmbeddingClient:
    """Test OpenAI embedding client."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock(spec=Config)
        config.get_embeddings_config.return_value = {
            "model": {
                "name": "text-embedding-3-small",
                "dimensions": 1536
            },
            "performance": {
                "request_timeout": 30,
                "rate_limit_delay": 0.1
            },
            "processing": {
                "max_retries": 3,
                "retry_failed": True
            },
            "usage": {
                "track_tokens": True,
                "log_requests": False
            }
        }
        return config
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10
        return mock_response
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_client_initialization_success(self, mock_config):
        """Test successful client initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                assert client.model == "text-embedding-3-small"
                assert client.model_dimensions == 1536
                assert client.max_retries == 3
                assert client.track_tokens is True
                
    def test_client_initialization_no_openai_library(self):
        """Test client initialization when OpenAI library is not available."""
        with patch("app.openai_client.OPENAI_AVAILABLE", False):
            with pytest.raises(OpenAIClientError, match="OpenAI library not installed"):
                OpenAIEmbeddingClient()
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_client_initialization_no_api_key(self, mock_config):
        """Test client initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY environment variable not set"):
                OpenAIEmbeddingClient(config=mock_config)
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_client_initialization_connection_failure(self, mock_config):
        """Test client initialization with connection failure."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.side_effect = Exception("Connection failed")
                
                with pytest.raises(OpenAIClientError, match="Connection validation failed"):
                    OpenAIEmbeddingClient(config=mock_config)
                    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_success(self, mock_config, mock_openai_response):
        """Test successful single embedding generation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Mock validation call
                mock_client.embeddings.create.return_value = mock_openai_response
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # Reset mock to track actual embedding call
                mock_client.embeddings.create.reset_mock()
                mock_client.embeddings.create.return_value = mock_openai_response
                
                result = client.generate_embedding("test text")
                
                assert isinstance(result, EmbeddingResult)
                assert result.text == "test text"
                assert result.model == "text-embedding-3-small"
                assert result.token_count == 10
                assert len(result.embedding) == 1536
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_empty_text(self, mock_config):
        """Test embedding generation with empty text."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                with pytest.raises(OpenAIClientError, match="Text cannot be empty"):
                    client.generate_embedding("")
                    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embeddings_batch_success(self, mock_config, mock_openai_response):
        """Test successful batch embedding generation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Mock validation call
                mock_client.embeddings.create.return_value = mock_openai_response
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # Mock batch response
                batch_response = Mock()
                batch_response.data = [Mock(), Mock()]
                batch_response.data[0].embedding = [0.1] * 1536
                batch_response.data[1].embedding = [0.2] * 1536
                batch_response.usage = Mock()
                batch_response.usage.total_tokens = 20
                
                mock_client.embeddings.create.reset_mock()
                mock_client.embeddings.create.return_value = batch_response
                
                texts = ["text1", "text2"]
                results = client.generate_embeddings(texts)
                
                assert len(results) == 2
                assert all(isinstance(r, EmbeddingResult) for r in results)
                assert results[0].text == "text1"
                assert results[1].text == "text2"
                assert results[0].token_count == 10  # 20 total / 2 texts
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embeddings_empty_list(self, mock_config):
        """Test batch embedding generation with empty list."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                results = client.generate_embeddings([])
                assert results == []
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embeddings_filter_empty_texts(self, mock_config, mock_openai_response):
        """Test batch embedding generation filters empty texts."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = mock_openai_response
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                with pytest.raises(OpenAIClientError, match="No valid texts provided"):
                    client.generate_embeddings(["", "   ", None])
                    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_with_retry(self, mock_config):
        """Test embedding generation with retry logic."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # First call for validation succeeds
                validation_response = Mock()
                validation_response.data = [Mock()]
                
                # Second call fails, third succeeds
                success_response = Mock()
                success_response.data = [Mock()]
                success_response.data[0].embedding = [0.1] * 1536
                success_response.usage = Mock()
                success_response.usage.total_tokens = 5
                
                mock_client.embeddings.create.side_effect = [
                    validation_response,  # validation call
                    Exception("First attempt fails"),  # first embedding call
                    success_response  # second embedding call succeeds
                ]
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                result = client.generate_embedding("test text")
                assert isinstance(result, EmbeddingResult)
                assert result.text == "test text"
                
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_max_retries_exceeded(self, mock_config):
        """Test embedding generation when max retries exceeded."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Validation succeeds
                validation_response = Mock()
                validation_response.data = [Mock()]
                mock_client.embeddings.create.return_value = validation_response
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # All embedding attempts fail
                mock_client.embeddings.create.side_effect = Exception("API Error")
                
                with pytest.raises(OpenAIClientError, match="Failed to generate embedding after 4 attempts"):
                    client.generate_embedding("test text")
                    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_get_model_info(self, mock_config):
        """Test getting model information."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                info = client.get_model_info()
                
                assert info["model_name"] == "text-embedding-3-small"
                assert info["dimensions"] == 1536
                assert info["max_retries"] == 3
                assert info["track_tokens"] is True
                
    def test_get_embedding_client_convenience_function(self):
        """Test the convenience function for getting a client."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAIEmbeddingClient") as mock_client_class:
                mock_instance = Mock()
                mock_client_class.return_value = mock_instance
                
                config = Mock()
                result = get_embedding_client(config=config)
                
                mock_client_class.assert_called_once_with(config=config)
                assert result == mock_instance
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_is_retryable_error_rate_limit(self, mock_config):
        """Test retryable error detection for rate limits."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # Rate limit errors should be retryable
                rate_limit_error = Exception("Rate limit exceeded (429)")
                assert client._is_retryable_error(rate_limit_error) is True
                
                timeout_error = Exception("Request timed out")
                assert client._is_retryable_error(timeout_error) is True
                
                connection_error = Exception("Connection failed")
                assert client._is_retryable_error(connection_error) is True
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_is_retryable_error_non_retryable(self, mock_config):
        """Test retryable error detection for non-retryable errors."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                mock_client.embeddings.create.return_value = Mock(data=[Mock()])
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # Authentication errors should not be retryable
                auth_error = Exception("Unauthorized (401)")
                assert client._is_retryable_error(auth_error) is False
                
                invalid_key_error = Exception("Invalid API key")
                assert client._is_retryable_error(invalid_key_error) is False
                
                bad_request_error = Exception("Bad request (400)")
                assert client._is_retryable_error(bad_request_error) is False
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_retry_on_retryable_error(self, mock_config):
        """Test embedding generation retries on retryable errors."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Validation succeeds
                validation_response = Mock()
                validation_response.data = [Mock()]
                
                # First call fails with retryable error, second succeeds
                success_response = Mock()
                success_response.data = [Mock()]
                success_response.data[0].embedding = [0.1] * 1536
                success_response.usage = Mock()
                success_response.usage.total_tokens = 5
                
                mock_client.embeddings.create.side_effect = [
                    validation_response,  # validation call
                    Exception("Rate limit exceeded (429)"),  # first embedding call fails
                    success_response  # second embedding call succeeds
                ]
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                with patch.object(client, '_is_retryable_error', return_value=True):
                    result = client.generate_embedding("test text")
                    assert isinstance(result, EmbeddingResult)
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embedding_no_retry_on_non_retryable_error(self, mock_config):
        """Test embedding generation doesn't retry on non-retryable errors."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Validation succeeds
                validation_response = Mock()
                validation_response.data = [Mock()]
                mock_client.embeddings.create.return_value = validation_response
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                # Embedding call fails with non-retryable error
                mock_client.embeddings.create.side_effect = Exception("Invalid API key (401)")
                
                with patch.object(client, '_is_retryable_error', return_value=False):
                    with pytest.raises(OpenAIClientError, match="Failed to generate embedding after 1 attempts"):
                        client.generate_embedding("test text")
    
    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI library not available")
    def test_generate_embeddings_batch_retry_logic(self, mock_config):
        """Test batch embedding generation retry logic."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("app.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Validation succeeds
                validation_response = Mock()
                validation_response.data = [Mock()]
                
                # First call fails, second succeeds
                success_response = Mock()
                success_response.data = [Mock(), Mock()]
                success_response.data[0].embedding = [0.1] * 1536
                success_response.data[1].embedding = [0.2] * 1536
                success_response.usage = Mock()
                success_response.usage.total_tokens = 20
                
                mock_client.embeddings.create.side_effect = [
                    validation_response,  # validation call
                    Exception("Rate limit exceeded (429)"),  # first batch call fails
                    success_response  # second batch call succeeds
                ]
                
                client = OpenAIEmbeddingClient(config=mock_config)
                
                with patch.object(client, '_is_retryable_error', return_value=True):
                    results = client.generate_embeddings(["text1", "text2"])
                    assert len(results) == 2
                    assert all(isinstance(r, EmbeddingResult) for r in results)