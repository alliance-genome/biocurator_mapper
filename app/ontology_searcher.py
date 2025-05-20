from typing import List, Dict
import asyncio
import logging

import openai
import weaviate

from .config import OPENAI_API_KEY, DEFAULT_K
from .ontology_manager import OntologyManager


class OntologySearcher:
    def __init__(self, manager: OntologyManager, openai_api_key: str = OPENAI_API_KEY):
        self.manager = manager
        openai.api_key = openai_api_key
        self.logger = logging.getLogger(__name__)

    async def _embed_passage(self, passage: str) -> List[float]:
        response = await openai.embeddings.create(input=passage, model="text-embedding-ada-002")
        return response.data[0].embedding

    async def search_ontology(
        self, passage: str, ontology_collection: str, k: int = DEFAULT_K
    ) -> List[Dict]:
        client = await self.manager.get_weaviate_client()
        embedding = await self._embed_passage(passage)
        try:
            result = await asyncio.to_thread(
                lambda: client.query.get(ontology_collection, ["id", "name", "definition"])
                .with_near_vector({"vector": embedding})
                .with_limit(k)
                .do()
            )
        except weaviate.exceptions.WeaviateBaseError as e:
            self.logger.exception(
                f"Weaviate query failed for collection {ontology_collection}"
            )
            return []
        hits = result.get("data", {}).get("Get", {}).get(ontology_collection, [])
        candidates = []
        for hit in hits:
            candidates.append(
                {
                    "id": hit.get("id"),
                    "name": hit.get("name"),
                    "definition": hit.get("definition"),
                }
            )
        return candidates
