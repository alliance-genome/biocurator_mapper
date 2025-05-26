import os
import logging
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

try:
    from openai import OpenAI
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    httpx = None

from app.config import Config

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    text: str
    token_count: int
    model: str


class OpenAIClientError(Exception):
    """Custom exception for OpenAI client errors."""
    pass


class OpenAIEmbeddingClient:
    def __init__(self, config: Optional[Config] = None):
        """Initialize OpenAI client.
        
        Args:
            config: Configuration object. If None, creates a new one.
            
        Raises:
            OpenAIClientError: If OpenAI library is not available or API key is missing.
        """
        if not OPENAI_AVAILABLE:
            raise OpenAIClientError(
                "OpenAI library not installed. Install with: pip install openai"
            )
            
        self.config = config or Config()
        self.embeddings_config = self.config.get_embeddings_config()
        
        # Get API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAIClientError(
                "OPENAI_API_KEY environment variable not set"
            )
            
        # Get model configuration
        model_config = self.embeddings_config.get("model", {})
        self.model = model_config.get("name", "text-embedding-3-small")
        self.model_dimensions = model_config.get("dimensions", 1536)
        
        # Performance settings
        performance_config = self.embeddings_config.get("performance", {})
        self.rate_limit_delay = performance_config.get("rate_limit_delay", 0.1)
        self.request_timeout = performance_config.get("request_timeout", 30)
        
        # Processing settings
        processing_config = self.embeddings_config.get("processing", {})
        self.max_retries = processing_config.get("max_retries", 3)
        self.retry_failed = processing_config.get("retry_failed", True)
        
        # Usage tracking
        usage_config = self.embeddings_config.get("usage", {})
        self.track_tokens = usage_config.get("track_tokens", True)
        self.log_requests = usage_config.get("log_requests", False)
        
        # Initialize client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=httpx.Timeout(60.0, read=30.0, write=30.0, connect=5.0) if httpx else self.request_timeout
            )
            self._validate_connection()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            raise OpenAIClientError(f"Failed to initialize OpenAI client: {e}")
    
    def _validate_connection(self):
        """Validate OpenAI API connection by performing a minimal test.
        
        Raises:
            OpenAIClientError: If connection validation fails.
        """
        try:
            # Attempt a minimal embedding to test connection
            response = self.client.embeddings.create(
                input=["OpenAI connection test"],
                model=self.model,
                dimensions=self.model_dimensions
            )
            
            if response.data and len(response.data) > 0:
                logger.info(f"Successfully validated OpenAI API connection using {self.model}")
            else:
                raise OpenAIClientError("No data returned from validation test")
                
        except Exception as e:
            logger.error(f"OpenAI API connection validation failed: {e}")
            raise OpenAIClientError(f"Connection validation failed: {e}")
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: Exception to check.
            
        Returns:
            True if error should be retried.
        """
        error_str = str(error).lower()
        
        # Rate limit errors are retryable
        if "rate limit" in error_str or "429" in error_str:
            return True
            
        # Timeout errors are retryable
        if "timeout" in error_str or "timed out" in error_str:
            return True
            
        # Connection errors are retryable
        if "connection" in error_str or "network" in error_str:
            return True
            
        # Server errors (5xx) are retryable
        if "server error" in error_str or "50" in error_str:
            return True
            
        # Service unavailable
        if "service unavailable" in error_str or "503" in error_str:
            return True
            
        # Client errors (4xx) except rate limits are generally not retryable
        if "400" in error_str or "401" in error_str or "403" in error_str or "404" in error_str:
            return False
            
        # Authentication errors are not retryable
        if "unauthorized" in error_str or "invalid api key" in error_str:
            return False
            
        # Default to retryable for unknown errors
        return True
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            EmbeddingResult containing the embedding and metadata.
            
        Raises:
            OpenAIClientError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise OpenAIClientError("Text cannot be empty")
            
        for attempt in range(self.max_retries + 1):
            try:
                if self.log_requests:
                    logger.debug(f"Generating embedding for text: {text[:100]}...")
                    
                # Apply rate limiting
                if attempt > 0:
                    delay = self.rate_limit_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
                elif self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                    
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model,
                    dimensions=self.model_dimensions
                )
                
                if not response.data or len(response.data) == 0:
                    raise OpenAIClientError("No embedding data returned from API")
                    
                embedding_data = response.data[0]
                token_count = response.usage.total_tokens if response.usage else 0
                
                if self.track_tokens:
                    logger.debug(f"Generated embedding with {token_count} tokens")
                    
                return EmbeddingResult(
                    embedding=embedding_data.embedding,
                    text=text,
                    token_count=token_count,
                    model=self.model
                )
                
            except Exception as e:
                # Handle specific error types
                error_msg = str(e)
                is_retryable = self._is_retryable_error(e)
                
                if attempt < self.max_retries and self.retry_failed and is_retryable:
                    logger.warning(f"Embedding generation attempt {attempt + 1} failed (retryable): {e}")
                    continue
                else:
                    if not is_retryable:
                        logger.error(f"Non-retryable error in embedding generation: {e}")
                    raise OpenAIClientError(f"Failed to generate embedding after {attempt + 1} attempts: {e}")

    def generate_embeddings(
        self, 
        texts: List[str], 
        dimensions: Optional[int] = None
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            dimensions: Override dimensions (uses config default if None).
            
        Returns:
            List of EmbeddingResult objects.
            
        Raises:
            OpenAIClientError: If batch embedding generation fails.
        """
        if not texts:
            return []
            
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise OpenAIClientError("No valid texts provided for embedding")
        
        # Use provided dimensions or config default
        embed_dimensions = dimensions or self.model_dimensions
            
        for attempt in range(self.max_retries + 1):
            try:
                if self.log_requests:
                    logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
                    
                # Apply rate limiting
                if attempt > 0:
                    delay = self.rate_limit_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(delay)
                elif self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                    
                response = self.client.embeddings.create(
                    input=valid_texts,
                    model=self.model,
                    dimensions=embed_dimensions
                )
                
                if not response.data or len(response.data) != len(valid_texts):
                    raise OpenAIClientError(
                        f"Expected {len(valid_texts)} embeddings, got {len(response.data) if response.data else 0}"
                    )
                    
                results = []
                token_count = response.usage.total_tokens if response.usage else 0
                
                for i, embedding_data in enumerate(response.data):
                    results.append(EmbeddingResult(
                        embedding=embedding_data.embedding,
                        text=valid_texts[i],
                        token_count=token_count // len(valid_texts),  # Approximate per-text token count
                        model=self.model
                    ))
                    
                if self.track_tokens:
                    logger.debug(f"Generated {len(results)} embeddings with {token_count} total tokens")
                    
                return results
                
            except Exception as e:
                # Handle specific error types
                is_retryable = self._is_retryable_error(e)
                
                if attempt < self.max_retries and self.retry_failed and is_retryable:
                    logger.warning(f"Batch embedding generation attempt {attempt + 1} failed (retryable): {e}")
                    continue
                else:
                    if not is_retryable:
                        logger.error(f"Non-retryable error in batch embedding generation: {e}")
                    raise OpenAIClientError(f"Failed to generate batch embeddings after {attempt + 1} attempts: {e}")
                    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model,
            "dimensions": self.model_dimensions,
            "max_retries": self.max_retries,
            "rate_limit_delay": self.rate_limit_delay,
            "request_timeout": self.request_timeout,
            "track_tokens": self.track_tokens
        }


def get_embedding_client(config: Optional[Config] = None) -> OpenAIEmbeddingClient:
    """Convenience function to get an OpenAI embedding client.
    
    Args:
        config: Configuration object. If None, creates a new one.
    
    Returns:
        Configured OpenAI embedding client.
    """
    return OpenAIEmbeddingClient(config=config)