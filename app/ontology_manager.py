import asyncio
import logging
import time
from typing import Any, Callable, Optional

import weaviate
import weaviate.classes as wvc
from openai import APIError, RateLimitError
from weaviate.exceptions import WeaviateBaseError

from .config import EMBEDDINGS_CONFIG, WEAVIATE_API_KEY, WEAVIATE_URL
from .config_updater import ConfigUpdater
from .go_parser import parse_enhanced_go_term


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

    def _extract_enhanced_term_data(self, raw_term: dict) -> dict:
        """Extract all useful fields from GO/DO JSON for better semantic matching."""
        # If the term is already parsed (has exact_synonyms, etc.), use it directly
        if "exact_synonyms" in raw_term:
            return {
                "term_id": raw_term["id"],
                "name": raw_term["name"],
                "definition": raw_term.get("definition", ""),
                "exact_synonyms": raw_term.get("exact_synonyms", []),
                "narrow_synonyms": raw_term.get("narrow_synonyms", []),
                "broad_synonyms": raw_term.get("broad_synonyms", []),
                "all_synonyms": raw_term.get("all_synonyms", []),
                "searchable_text": raw_term.get("searchable_text", "")
            }

        # Check if this is a raw GO/DO node (has 'lbl' and 'meta' structure)
        if "lbl" in raw_term and "meta" in raw_term:
            # Use the GO parser which handles both GO and DO formats
            parsed_term = parse_enhanced_go_term(raw_term)
            if parsed_term:
                return {
                    "term_id": parsed_term["id"],
                    "name": parsed_term["name"],
                    "definition": parsed_term.get("definition", ""),
                    "exact_synonyms": parsed_term.get("exact_synonyms", []),
                    "narrow_synonyms": parsed_term.get("narrow_synonyms", []),
                    "broad_synonyms": parsed_term.get("broad_synonyms", []),
                    "all_synonyms": parsed_term.get("all_synonyms", []),
                    "searchable_text": ""  # Will be built by _build_searchable_text
                }

        # Fallback: extract basic fields from other formats
        term_data = {
            "term_id": raw_term.get("id", ""),
            "name": raw_term.get("name", ""),
            "definition": raw_term.get("definition", "")
        }

        # Initialize empty synonym arrays for unparsed data
        term_data.update({
            "exact_synonyms": [],
            "narrow_synonyms": [],
            "broad_synonyms": [],
            "all_synonyms": [],
            "searchable_text": ""  # Will be built by _build_searchable_text
        })

        return term_data

    def _build_searchable_text(self, term_data: dict) -> str:
        """Build comprehensive text for semantic search based on configuration."""
        vectorize_fields = EMBEDDINGS_CONFIG.get("vectorize_fields", {})
        preprocessing = EMBEDDINGS_CONFIG.get("preprocessing", {})

        components = []

        # Add fields based on configuration
        if vectorize_fields.get("name", True):
            name = term_data.get("name", "")
            if name:
                components.append(name)

        if vectorize_fields.get("definition", True):
            definition = term_data.get("definition", "")
            if definition:
                components.append(definition)

        if vectorize_fields.get("synonyms", True):
            # Add all synonyms for richer semantic content
            components.extend(term_data.get("exact_synonyms", []))
            components.extend(term_data.get("narrow_synonyms", []))
            components.extend(term_data.get("broad_synonyms", []))

        # Apply preprocessing
        if preprocessing.get("lowercase", False):
            components = [c.lower() for c in components if c]

        if preprocessing.get("remove_punctuation", False):
            import string
            translator = str.maketrans('', '', string.punctuation)
            components = [c.translate(translator) for c in components if c]

        # Join with configured separator
        separator = preprocessing.get("combine_fields_separator", " | ")
        return separator.join(filter(None, components))

    async def create_and_load_ontology_collection(
        self,
        collection_name: str,
        ontology_terms: list[dict],
        openai_api_key: str,
        progress_callback: Optional[Callable[[str, int, str, dict[str, Any]], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None
    ) -> None:
        """Create Weaviate collection with richer ontology data and progress tracking.

        Args:
            collection_name: Name of the collection to create
            ontology_terms: List of ontology terms to load
            openai_api_key: OpenAI API key for embeddings
            progress_callback: Optional callback for progress updates
                              (status, percentage, message, extra_data)
            cancellation_check: Optional callback to check if operation should be cancelled
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

        # Check for cancellation
        if cancellation_check and cancellation_check():
            update_progress("cancelled", 0, "Operation cancelled by user")
            return

        # Delete existing collection if it exists
        try:
            await asyncio.to_thread(client.collections.delete, collection_name)
            self.logger.info(f"Deleted existing collection: {collection_name}")
            update_progress("initializing", 5, f"Deleted existing collection: {collection_name}")
        except WeaviateBaseError as e:
            if "not found" not in str(e).lower():
                self.logger.error(f"Error deleting collection: {e}")
                update_progress("error", 0, f"Failed to delete existing collection: {str(e)}")
                raise
        except Exception as e:
            self.logger.warning(f"Collection might not exist: {e}")

        # Check for cancellation before creating collection
        if cancellation_check and cancellation_check():
            update_progress("cancelled", 0, "Operation cancelled by user")
            return

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
            # Check for cancellation during processing
            if cancellation_check and cancellation_check():
                update_progress("cancelled", 0, "Operation cancelled by user during term processing")
                return

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
        failed_batches = []

        for batch_idx in range(0, len(enhanced_terms), batch_size):
            batch_terms = enhanced_terms[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1

            # Calculate progress (45-95% for embedding generation)
            progress_percentage = 45 + int((batch_idx / len(enhanced_terms)) * 50)

            # Check for cancellation before each batch
            if cancellation_check and cancellation_check():
                update_progress("cancelled", progress_percentage, "Operation cancelled by user during batch processing")
                return

            update_progress(
                "embedding_batch",
                progress_percentage,
                f"Processing batch {batch_num}/{total_batches} ({len(batch_terms)} terms)",
                current_batch=batch_num,
                batch_size=len(batch_terms)
            )

            # Retry logic for batch processing
            batch_retry_count = 0
            batch_success = False

            while batch_retry_count <= max_retries and not batch_success:
                try:
                    # Use v4 batch API with error handling
                    batch_failed_terms = []

                    with collection.batch.dynamic() as batch:
                        for idx, term in enumerate(batch_terms):
                            # Check for cancellation every 10 terms within a batch
                            if idx > 0 and idx % 10 == 0 and cancellation_check and cancellation_check():
                                # Cancel the batch gracefully
                                update_progress("cancelled", progress_percentage, "Operation cancelled by user during term processing")
                                return

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
                            except Exception as e:
                                self.logger.error(f"Failed to add term {term.get('term_id')}: {e}")
                                batch_failed_terms.append(term)
                                embedding_stats["failed_terms"] += 1

                    # If some terms succeeded, count the batch as partially successful
                    successful_terms = len(batch_terms) - len(batch_failed_terms)
                    embedding_stats["processed_terms"] += successful_terms

                    if len(batch_failed_terms) == 0:
                        batch_success = True
                        embedding_stats["batches_completed"] += 1
                    else:
                        # Check for cancellation before retry
                        if cancellation_check and cancellation_check():
                            update_progress("cancelled", progress_percentage, "Operation cancelled by user before retry")
                            return

                        # Retry only failed terms
                        if retry_failed and batch_retry_count < max_retries:
                            batch_terms = batch_failed_terms
                            batch_retry_count += 1
                            embedding_stats["retry_count"] += 1
                            update_progress(
                                "retrying_batch",
                                progress_percentage,
                                f"Retrying {len(batch_failed_terms)} failed terms from batch {batch_num}"
                            )
                            await asyncio.sleep(rate_limit_delay * 2)  # Double delay for retries
                            continue
                        else:
                            batch_success = True  # Move on even with failures
                            embedding_stats["batches_completed"] += 1

                    # Add rate limiting delay
                    if rate_limit_delay > 0 and batch_idx + batch_size < len(enhanced_terms):
                        await asyncio.sleep(rate_limit_delay)

                except (RateLimitError, APIError) as e:
                    # Handle OpenAI API errors
                    self.logger.error(f"OpenAI API error in batch {batch_num}: {e}")

                    # Check for cancellation before retry
                    if cancellation_check and cancellation_check():
                        update_progress("cancelled", progress_percentage, "Operation cancelled by user during rate limit handling")
                        return

                    if batch_retry_count < max_retries:
                        batch_retry_count += 1
                        embedding_stats["retry_count"] += 1
                        wait_time = min(rate_limit_delay * (2 ** batch_retry_count), 60)  # Exponential backoff
                        update_progress(
                            "rate_limited",
                            progress_percentage,
                            f"Rate limited on batch {batch_num}, waiting {wait_time:.1f}s before retry..."
                        )

                        # Check for cancellation during wait with 1-second intervals
                        for _ in range(int(wait_time)):
                            if cancellation_check and cancellation_check():
                                update_progress("cancelled", progress_percentage, "Operation cancelled by user during rate limit wait")
                                return
                            await asyncio.sleep(1)
                        await asyncio.sleep(wait_time % 1)  # Sleep for remaining fraction
                    else:
                        failed_batches.append((batch_num, str(e)))
                        update_progress(
                            "batch_error",
                            progress_percentage,
                            f"Batch {batch_num} failed after {max_retries} retries: {str(e)}"
                        )
                        break

                except WeaviateBaseError as e:
                    # Handle Weaviate errors
                    self.logger.error(f"Weaviate error in batch {batch_num}: {e}")

                    # Check for cancellation before retry
                    if cancellation_check and cancellation_check():
                        update_progress("cancelled", progress_percentage, "Operation cancelled by user during Weaviate error handling")
                        return

                    if batch_retry_count < max_retries:
                        batch_retry_count += 1
                        embedding_stats["retry_count"] += 1
                        update_progress(
                            "weaviate_error",
                            progress_percentage,
                            f"Weaviate error on batch {batch_num}, retrying..."
                        )
                        await asyncio.sleep(rate_limit_delay * 2)
                    else:
                        failed_batches.append((batch_num, str(e)))
                        update_progress(
                            "batch_error",
                            progress_percentage,
                            f"Batch {batch_num} failed with Weaviate error: {str(e)}"
                        )
                        break

                except Exception as e:
                    # Handle unexpected errors
                    self.logger.error(f"Unexpected error in batch {batch_num}: {e}")
                    failed_batches.append((batch_num, str(e)))
                    update_progress(
                        "batch_error",
                        progress_percentage,
                        f"Batch {batch_num} encountered unexpected error: {str(e)}"
                    )
                    break

        # Calculate final statistics
        elapsed_time = time.time() - embedding_stats["start_time"]
        terms_per_second = embedding_stats["processed_terms"] / elapsed_time if elapsed_time > 0 else 0

        # Determine final status
        if cancellation_check and cancellation_check():
            final_status = "cancelled"
            final_message = f"Operation cancelled. Processed {embedding_stats['processed_terms']}/{embedding_stats['total_terms']} terms before cancellation"
        elif len(failed_batches) > 0:
            final_status = "completed_with_errors"
            final_message = (
                f"Completed with errors: {embedding_stats['processed_terms']} terms imported, "
                f"{embedding_stats['failed_terms']} failed, {len(failed_batches)} batches had errors"
            )
        elif embedding_stats["failed_terms"] > 0:
            final_status = "completed_with_failures"
            final_message = (
                f"Completed with failures: {embedding_stats['processed_terms']} terms imported, "
                f"{embedding_stats['failed_terms']} terms failed"
            )
        else:
            final_status = "completed"
            final_message = f"Successfully imported all {embedding_stats['processed_terms']} terms in {elapsed_time:.1f}s"

        update_progress(
            final_status,
            100,
            final_message,
            elapsed_time=elapsed_time,
            terms_per_second=terms_per_second,
            failed_batches=failed_batches
        )

        self.logger.info(
            f"Embedding generation {final_status}: "
            f"{embedding_stats['processed_terms']} processed, "
            f"{embedding_stats['failed_terms']} failed, "
            f"{embedding_stats['retry_count']} retries, "
            f"{elapsed_time:.1f}s elapsed, "
            f"{terms_per_second:.1f} terms/s"
        )

