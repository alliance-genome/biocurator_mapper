from typing import List, Dict

import openai

from .config import OPENAI_API_KEY, DEFAULT_K
from .ontology_manager import OntologyManager


class OntologySearcher:
    def __init__(self, manager: OntologyManager, openai_api_key: str = OPENAI_API_KEY):
        self.manager = manager
        openai.api_key = openai_api_key

    def _embed_passage(self, passage: str) -> List[float]:
        response = openai.Embedding.create(input=passage, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]

    def search_ontology(
        self, passage: str, ontology_collection: str, k: int = DEFAULT_K
    ) -> List[Dict]:
        client = self.manager.get_weaviate_client()
        embedding = self._embed_passage(passage)
        result = (
            client.query.get(ontology_collection, ["id", "name", "definition"])
            .with_near_vector({"vector": embedding})
            .with_limit(k)
            .do()
        )
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
