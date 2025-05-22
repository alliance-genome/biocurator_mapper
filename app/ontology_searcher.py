from typing import List, Dict
import asyncio
import logging

import openai
import weaviate

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
            # Enhanced query that retrieves all stored fields
            result = await asyncio.to_thread(
                lambda: client.query.get(
                    ontology_collection, 
                    [
                        "id", 
                        "name", 
                        "definition",
                        "exact_synonyms",
                        "narrow_synonyms", 
                        "broad_synonyms",
                        "all_synonyms",
                        "cross_references",
                        "namespace"
                    ]
                )
                .with_near_vector({"vector": embedding})
                .with_limit(k)
                .with_additional(["distance", "certainty"])  # Include similarity scores
                .do()
            )
        except weaviate.exceptions.WeaviateBaseError as e:
            self.logger.exception(f"Weaviate query failed for collection {ontology_collection}")
            return []
        
        hits = result.get("data", {}).get("Get", {}).get(ontology_collection, [])
        candidates = []
        
        for hit in hits:
            # Get similarity metadata
            additional = hit.get("_additional", {})
            distance = additional.get("distance", 1.0)
            certainty = additional.get("certainty", 0.0)
            
            candidate = {
                "id": hit.get("id"),
                "name": hit.get("name"),
                "definition": hit.get("definition"),
                "exact_synonyms": hit.get("exact_synonyms", []),
                "narrow_synonyms": hit.get("narrow_synonyms", []),
                "broad_synonyms": hit.get("broad_synonyms", []),
                "all_synonyms": hit.get("all_synonyms", []),
                "cross_references": hit.get("cross_references", []),
                "namespace": hit.get("namespace", ""),
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
            query = client.query.get(
                ontology_collection,
                [
                    "id", "name", "definition", "exact_synonyms", 
                    "narrow_synonyms", "broad_synonyms", "all_synonyms",
                    "cross_references", "namespace"
                ]
            ).with_near_vector({"vector": embedding}).with_limit(k)
            
            # Add namespace filter if specified
            if namespace_filter:
                query = query.with_where({
                    "path": ["namespace"],
                    "operator": "Equal",
                    "valueText": namespace_filter
                })
            
            result = await asyncio.to_thread(query.do)
            
        except weaviate.exceptions.WeaviateBaseError as e:
            self.logger.exception(f"Enhanced filtered query failed: {e}")
            return []
        
        hits = result.get("data", {}).get("Get", {}).get(ontology_collection, [])
        candidates = []
        
        for hit in hits:
            candidate = {
                "id": hit.get("id"),
                "name": hit.get("name"),
                "definition": hit.get("definition"),
                "exact_synonyms": hit.get("exact_synonyms", []),
                "narrow_synonyms": hit.get("narrow_synonyms", []),
                "broad_synonyms": hit.get("broad_synonyms", []),
                "all_synonyms": hit.get("all_synonyms", []),
                "cross_references": hit.get("cross_references", []),
                "namespace": hit.get("namespace", "")
            }
            candidates.append(candidate)
        
        return candidates