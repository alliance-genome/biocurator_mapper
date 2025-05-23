from typing import List, Dict
import asyncio
import logging

import openai
import weaviate
import weaviate.classes as wvc

from .config import OPENAI_API_KEY, DEFAULT_K
from .ontology_manager import OntologyManager


class OntologySearcher:
    """Enhanced ontology searcher that leverages richer GO term data."""
    
    def __init__(self, manager: OntologyManager, openai_api_key: str = OPENAI_API_KEY):
        self.manager = manager
        openai.api_key = openai_api_key
        self.logger = logging.getLogger(__name__)

    async def _embed_passage(self, passage: str) -> List[float]:
        """Create embedding for input passage."""
        response = await openai.embeddings.create(
            input=passage, 
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    async def search_ontology(
        self, passage: str, ontology_collection: str, k: int = DEFAULT_K
    ) -> List[Dict]:
        """
        Semantic search that returns richer term data.
        
        The search is performed against the 'searchable_text' field which contains
        name + definition + all synonyms for better semantic matching.
        """
        client = await self.manager.get_weaviate_client()
        embedding = await self._embed_passage(passage)
        
        try:
            # Get the collection
            collection = client.collections.get(ontology_collection)
            
            # Enhanced query using v4 API
            response = await asyncio.to_thread(
                lambda: collection.query.near_vector(
                    near_vector=embedding,
                    limit=k,
                    return_properties=[
                        "term_id", 
                        "name", 
                        "definition",
                        "exact_synonyms",
                        "narrow_synonyms", 
                        "broad_synonyms",
                        "all_synonyms"
                    ],
                    return_metadata=["distance", "certainty"]
                )
            )
            
        except Exception as e:
            self.logger.exception(f"Weaviate query failed for collection {ontology_collection}")
            return []
        
        candidates = []
        
        for obj in response.objects:
            # Get similarity metadata
            distance = obj.metadata.distance if obj.metadata.distance else 1.0
            certainty = obj.metadata.certainty if obj.metadata.certainty else 0.0
            
            candidate = {
                "id": obj.properties.get("term_id"),
                "name": obj.properties.get("name"),
                "definition": obj.properties.get("definition"),
                "exact_synonyms": obj.properties.get("exact_synonyms", []),
                "narrow_synonyms": obj.properties.get("narrow_synonyms", []),
                "broad_synonyms": obj.properties.get("broad_synonyms", []),
                "all_synonyms": obj.properties.get("all_synonyms", []),
                "cross_references": obj.properties.get("cross_references", []),
                "namespace": obj.properties.get("namespace", ""),
                "similarity_distance": distance,
                "similarity_certainty": certainty
            }
            candidates.append(candidate)
        
        self.logger.info(
            f"Found {len(candidates)} candidates for passage: '{passage[:50]}...'"
        )
        
        return candidates

    async def search_with_filters(
        self, 
        passage: str, 
        ontology_collection: str, 
        namespace_filter: str = None,
        k: int = DEFAULT_K
    ) -> List[Dict]:
        """
        Enhanced search with optional filtering by GO namespace 
        (biological_process, molecular_function, cellular_component).
        """
        client = await self.manager.get_weaviate_client()
        embedding = await self._embed_passage(passage)
        
        try:
            # Get the collection
            collection = client.collections.get(ontology_collection)
            
            # Build the query with optional filtering
            where_filter = None
            if namespace_filter:
                where_filter = wvc.query.Filter.by_property("namespace").equal(namespace_filter)
            
            response = await asyncio.to_thread(
                lambda: collection.query.near_vector(
                    near_vector=embedding,
                    limit=k,
                    where=where_filter,
                    return_properties=[
                        "term_id", "name", "definition", "exact_synonyms", 
                        "narrow_synonyms", "broad_synonyms", "all_synonyms",
                        "namespace"
                    ]
                )
            )
            
        except Exception as e:
            self.logger.exception(f"Enhanced filtered query failed: {e}")
            return []
        
        candidates = []
        
        for obj in response.objects:
            candidate = {
                "id": obj.properties.get("term_id"),
                "name": obj.properties.get("name"),
                "definition": obj.properties.get("definition"),
                "exact_synonyms": obj.properties.get("exact_synonyms", []),
                "narrow_synonyms": obj.properties.get("narrow_synonyms", []),
                "broad_synonyms": obj.properties.get("broad_synonyms", []),
                "all_synonyms": obj.properties.get("all_synonyms", []),
                "cross_references": obj.properties.get("cross_references", []),
                "namespace": obj.properties.get("namespace", "")
            }
            candidates.append(candidate)
        
        return candidates