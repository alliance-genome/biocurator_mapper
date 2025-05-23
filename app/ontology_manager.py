from typing import Optional, List, Dict
import asyncio
import logging

import weaviate
import weaviate.classes as wvc

from .config import WEAVIATE_URL, WEAVIATE_API_KEY
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
        self, collection_name: str, ontology_terms: List[Dict], openai_api_key: str
    ) -> None:
        """Create Weaviate collection with richer ontology data."""
        client = await self.get_weaviate_client()

        # Delete existing collection if it exists
        try:
            await asyncio.to_thread(client.collections.delete, collection_name)
            self.logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, which is fine
            pass

        # Create enhanced collection with synonym support
        self.logger.info(f"Creating enhanced collection: {collection_name}")
        
        # Create the collection with proper v4 API
        await asyncio.to_thread(
            client.collections.create,
            name=collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
                model="ada",
                model_version="002",
                type_="text"
            ),
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

        # Process and load enhanced term data
        self.logger.info(f"Processing {len(ontology_terms)} terms for enhanced storage")
        enhanced_terms = []
        
        for term in ontology_terms:
            enhanced_term = self._extract_enhanced_term_data(term)
            enhanced_term["searchable_text"] = self._build_searchable_text(enhanced_term)
            enhanced_terms.append(enhanced_term)

        # Batch import enhanced data
        self.logger.info(f"Starting enhanced batch import into {collection_name}")
        
        # Get the collection object
        collection = client.collections.get(collection_name)
        
        # Use v4 batch API
        with collection.batch.dynamic() as batch:
            for term in enhanced_terms:
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
        
        self.logger.info(f"Successfully imported enhanced data for {len(enhanced_terms)} terms")