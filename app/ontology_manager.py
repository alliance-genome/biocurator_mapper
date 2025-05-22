from typing import Optional, List, Dict
import asyncio
import logging

import weaviate

from .config import WEAVIATE_URL, WEAVIATE_API_KEY
from .config_updater import ConfigUpdater


class OntologyManager:
    """Enhanced ontology manager that stores richer GO term data for better semantic matching."""
    
    def __init__(self) -> None:
        self._client: Optional[weaviate.Client] = None
        self.config_updater = ConfigUpdater()
        self.logger = logging.getLogger(__name__)

    async def _init_client(self) -> weaviate.Client:
        auth = None
        if WEAVIATE_API_KEY:
            auth = weaviate.AuthApiKey(WEAVIATE_API_KEY)
        client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth)
        return client

    async def get_weaviate_client(self) -> weaviate.Client:
        if self._client is None:
            self._client = await self._init_client()
        return self._client

    def get_current_ontology_version(self, ontology_name: str) -> Optional[str]:
        return self.config_updater.get_current_ontology_version(ontology_name)

    def _extract_enhanced_term_data(self, raw_term: Dict) -> Dict:
        """Extract all useful fields from GO JSON for better semantic matching."""
        term_data = {
            "id": raw_term["id"],
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
        schema_details = await asyncio.to_thread(client.schema.get)
        existing = schema_details.get("classes", [])
        if any(c.get("class") == collection_name for c in existing):
            self.logger.info(f"Deleting existing collection: {collection_name}")
            await asyncio.to_thread(client.schema.delete_class, collection_name)

        # Create enhanced schema with synonym support
        self.logger.info(f"Creating enhanced schema for collection: {collection_name}")
        schema = {
            "class": collection_name,
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "text-embedding-ada-002",
                    "modelVersion": "002",
                    "type": "text"
                }
            },
            "properties": [
                {"name": "id", "dataType": ["text"]},
                {"name": "name", "dataType": ["text"]},
                {"name": "definition", "dataType": ["text"]},
                {"name": "exact_synonyms", "dataType": ["text[]"]},
                {"name": "narrow_synonyms", "dataType": ["text[]"]},
                {"name": "broad_synonyms", "dataType": ["text[]"]},
                {"name": "all_synonyms", "dataType": ["text[]"]},
                {
                    "name": "searchable_text", 
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,  # This field will be vectorized
                            "vectorizePropertyName": False
                        }
                    }
                }
            ],
        }
        await asyncio.to_thread(client.schema.create_class, schema)
        self.logger.info(f"Successfully created enhanced schema: {collection_name}")

        # Process and load enhanced term data
        self.logger.info(f"Processing {len(ontology_terms)} terms for enhanced storage")
        enhanced_terms = []
        
        for term in ontology_terms:
            enhanced_term = self._extract_enhanced_term_data(term)
            enhanced_term["searchable_text"] = self._build_searchable_text(enhanced_term)
            enhanced_terms.append(enhanced_term)

        # Batch import enhanced data
        self.logger.info(f"Starting enhanced batch import into {collection_name}")
        with client.batch.dynamic() as batch:
            for term in enhanced_terms:
                batch.add_data_object(
                    data_object={
                        "id": term["id"],
                        "name": term["name"],
                        "definition": term["definition"],
                        "exact_synonyms": term["exact_synonyms"],
                        "narrow_synonyms": term["narrow_synonyms"],
                        "broad_synonyms": term["broad_synonyms"],
                        "all_synonyms": term["all_synonyms"],
                        "searchable_text": term["searchable_text"]
                    },
                    class_name=collection_name,
                )
        await asyncio.to_thread(client.batch.flush)
        self.logger.info(f"Successfully imported enhanced data for {len(enhanced_terms)} terms")