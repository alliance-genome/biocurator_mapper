from typing import Optional, List, Dict, Callable, Any
import asyncio
import logging
import time
from datetime import datetime
import openai

import weaviate
import weaviate.classes as wvc

from .config import WEAVIATE_URL, WEAVIATE_API_KEY, EMBEDDINGS_CONFIG
from .config_updater import ConfigUpdater


class OntologyManager:
    """Enhanced ontology manager that stores richer GO term data for better semantic matching."""
    
    def __init__(self) -> None:
        self._client: Optional[weaviate.WeaviateClient] = None
        self.config_updater = ConfigUpdater()
        self.logger = logging.getLogger(__name__)

    async def _init_client(self) -> weaviate.WeaviateClient:
        # Extract host and port from URL
        url_parts = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080
        
        if WEAVIATE_API_KEY:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.Auth.api_key(WEAVIATE_API_KEY)
            )
        else:
            client = weaviate.connect_to_local(
                host=host,
                port=port
            )
        return client

    async def get_weaviate_client(self) -> weaviate.WeaviateClient:
        if self._client is None:
            self._client = await self._init_client()
        return self._client
    
    async def check_weaviate_health(self) -> bool:
        """Check if Weaviate is healthy and accessible."""
        try:
            client = await self.get_weaviate_client()
            # Check if Weaviate is ready
            return client.is_ready()
        except Exception as e:
            self.logger.error(f"Weaviate health check failed: {e}")
            return False

    def get_current_ontology_version(self, ontology_name: str) -> Optional[str]:
        return self.config_updater.get_current_ontology_version(ontology_name)

    def _extract_enhanced_term_data(self, raw_term: Dict) -> Dict:
        """Extract all useful fields from GO JSON for better semantic matching."""
        term_data = {
            "term_id": raw_term["id"],
            "name": raw_term["name"], 
            "definition": raw_term.get("definition", "")
        }
        
        # Extract synonyms for better matching
        synonyms = []
        exact_synonyms = []
        narrow_synonyms = []
        broad_synonyms = []
        
        # This would be populated during parsing from the actual GO JSON
        # For now, return basic structure
        term_data.update({
            "exact_synonyms": exact_synonyms,
            "narrow_synonyms": narrow_synonyms, 
            "broad_synonyms": broad_synonyms,
            "all_synonyms": synonyms,
            "searchable_text": ""  # Combined text for vectorization
        })
        
        return term_data

    def _build_searchable_text(self, term_data: Dict) -> str:
        """Build comprehensive text for semantic search."""
        components = [
            term_data.get("name", ""),
            term_data.get("definition", "")
        ]
        
        # Add all synonyms for richer semantic content
        components.extend(term_data.get("exact_synonyms", []))
        components.extend(term_data.get("narrow_synonyms", []))
        components.extend(term_data.get("broad_synonyms", []))
        
        return " ".join(filter(None, components))

    async def create_and_load_ontology_collection(
        self, 
        collection_name: str, 
        ontology_terms: List[Dict], 
        openai_api_key: str,
        progress_callback: Optional[Callable[[str, int, str, Dict[str, Any]], None]] = None
    ) -> None:
        """Create Weaviate collection with richer ontology data and progress tracking.
        
        Args:
            collection_name: Name of the collection to create
            ontology_terms: List of ontology terms to load
            openai_api_key: OpenAI API key for embeddings
            progress_callback: Optional callback for progress updates
                              (status, percentage, message, extra_data)
        """
        client = await self.get_weaviate_client()
        
        # Initialize embedding statistics
        embedding_stats = {
            "total_terms": len(ontology_terms),
            "processed_terms": 0,
            "failed_terms": 0,
            "retry_count": 0,
            "token_usage": 0,
            "start_time": time.time(),
            "batches_completed": 0,
            "total_batches": 0
        }
        
        def update_progress(status: str, percentage: int, message: str, **kwargs):
            """Update progress with additional data."""
            if progress_callback:
                extra_data = {**embedding_stats, **kwargs}
                progress_callback(status, percentage, message, extra_data)

        update_progress("initializing", 0, "Initializing embedding generation...")
        
        # Delete existing collection if it exists
        try:
            await asyncio.to_thread(client.collections.delete, collection_name)
            self.logger.info(f"Deleted existing collection: {collection_name}")
            update_progress("initializing", 5, f"Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, which is fine
            pass

        # Create enhanced collection with synonym support
        self.logger.info(f"Creating enhanced collection: {collection_name}")
        update_progress("creating_collection", 10, f"Creating collection: {collection_name}")
        
        # Get embedding model configuration
        embedding_model = EMBEDDINGS_CONFIG.get("model", {})
        model_name = embedding_model.get("name", "text-ada-002")
        
        # Map model names to Weaviate configuration
        if model_name == "text-ada-002":
            vectorizer_model = "ada"
            model_version = "002"
        elif model_name == "text-embedding-3-small":
            vectorizer_model = "text-embedding-3-small"
            model_version = None
        elif model_name == "text-embedding-3-large":
            vectorizer_model = "text-embedding-3-large" 
            model_version = None
        else:
            vectorizer_model = "ada"
            model_version = "002"
        
        # Create the collection with proper v4 API
        vectorizer_kwargs = {
            "model": vectorizer_model,
            "type_": "text"
        }
        if model_version:
            vectorizer_kwargs["model_version"] = model_version
            
        await asyncio.to_thread(
            client.collections.create,
            name=collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(**vectorizer_kwargs),
            properties=[
                wvc.config.Property(
                    name="term_id",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="name",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="definition",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="exact_synonyms",
                    data_type=wvc.config.DataType.TEXT_ARRAY
                ),
                wvc.config.Property(
                    name="narrow_synonyms",
                    data_type=wvc.config.DataType.TEXT_ARRAY
                ),
                wvc.config.Property(
                    name="broad_synonyms",
                    data_type=wvc.config.DataType.TEXT_ARRAY
                ),
                wvc.config.Property(
                    name="all_synonyms",
                    data_type=wvc.config.DataType.TEXT_ARRAY
                ),
                wvc.config.Property(
                    name="searchable_text",
                    data_type=wvc.config.DataType.TEXT,
                    skip_vectorization=False
                )
            ]
        )
        self.logger.info(f"Successfully created enhanced collection: {collection_name}")
        update_progress("created_collection", 15, "Collection created successfully")

        # Process and load enhanced term data
        self.logger.info(f"Processing {len(ontology_terms)} terms for enhanced storage")
        update_progress("processing_terms", 20, f"Processing {len(ontology_terms)} terms...")
        
        enhanced_terms = []
        
        for i, term in enumerate(ontology_terms):
            enhanced_term = self._extract_enhanced_term_data(term)
            enhanced_term["searchable_text"] = self._build_searchable_text(enhanced_term)
            enhanced_terms.append(enhanced_term)
            
            # Update progress every 100 terms
            if (i + 1) % 100 == 0:
                percentage = 20 + int((i + 1) / len(ontology_terms) * 20)  # 20-40%
                update_progress(
                    "processing_terms", 
                    percentage,
                    f"Processed {i + 1}/{len(ontology_terms)} terms"
                )

        update_progress("processing_complete", 40, f"Processed all {len(enhanced_terms)} terms")

        # Configure batch processing
        batch_config = EMBEDDINGS_CONFIG.get("processing", {})
        batch_size = batch_config.get("batch_size", 100)
        parallel_processing = batch_config.get("parallel_processing", True)
        retry_failed = batch_config.get("retry_failed", True)
        max_retries = batch_config.get("max_retries", 3)
        
        # Performance settings
        perf_config = EMBEDDINGS_CONFIG.get("performance", {})
        rate_limit_delay = perf_config.get("rate_limit_delay", 0.1)
        
        # Calculate total batches
        total_batches = (len(enhanced_terms) + batch_size - 1) // batch_size
        embedding_stats["total_batches"] = total_batches
        
        # Batch import enhanced data with progress tracking
        self.logger.info(f"Starting enhanced batch import into {collection_name}")
        update_progress(
            "embedding_generation", 
            45, 
            f"Starting embedding generation ({total_batches} batches, {batch_size} terms per batch)"
        )
        
        # Get the collection object
        collection = client.collections.get(collection_name)
        
        # Process terms in batches with detailed progress
        for batch_idx in range(0, len(enhanced_terms), batch_size):
            batch_terms = enhanced_terms[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            # Calculate progress (45-95% for embedding generation)
            progress_percentage = 45 + int((batch_idx / len(enhanced_terms)) * 50)
            
            update_progress(
                "embedding_batch",
                progress_percentage,
                f"Processing batch {batch_num}/{total_batches} ({len(batch_terms)} terms)",
                current_batch=batch_num,
                batch_size=len(batch_terms)
            )
            
            try:
                # Use v4 batch API with error handling
                with collection.batch.dynamic() as batch:
                    for term in batch_terms:
                        try:
                            batch.add_object({
                                "term_id": term["term_id"],
                                "name": term["name"],
                                "definition": term["definition"],
                                "exact_synonyms": term["exact_synonyms"],
                                "narrow_synonyms": term["narrow_synonyms"],
                                "broad_synonyms": term["broad_synonyms"],
                                "all_synonyms": term["all_synonyms"],
                                "searchable_text": term["searchable_text"]
                            })
                            embedding_stats["processed_terms"] += 1
                        except Exception as e:
                            self.logger.error(f"Failed to add term {term.get('term_id')}: {e}")
                            embedding_stats["failed_terms"] += 1
                            
                embedding_stats["batches_completed"] += 1
                
                # Add rate limiting delay
                if rate_limit_delay > 0 and batch_idx + batch_size < len(enhanced_terms):
                    await asyncio.sleep(rate_limit_delay)
                    
            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}")
                update_progress(
                    "batch_error",
                    progress_percentage,
                    f"Batch {batch_num} encountered an error, continuing...",
                    error=str(e)
                )
        
        # Calculate final statistics
        elapsed_time = time.time() - embedding_stats["start_time"]
        terms_per_second = embedding_stats["processed_terms"] / elapsed_time if elapsed_time > 0 else 0
        
        update_progress(
            "completed",
            100,
            f"Successfully imported {embedding_stats['processed_terms']} terms "
            f"({embedding_stats['failed_terms']} failed) in {elapsed_time:.1f}s",
            elapsed_time=elapsed_time,
            terms_per_second=terms_per_second
        )
        
        self.logger.info(
            f"Embedding generation complete: "
            f"{embedding_stats['processed_terms']} processed, "
            f"{embedding_stats['failed_terms']} failed, "
            f"{elapsed_time:.1f}s elapsed, "
            f"{terms_per_second:.1f} terms/s"
        )