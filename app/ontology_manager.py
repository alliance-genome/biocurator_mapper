from typing import Optional, List, Dict
import asyncio

import weaviate

from .config import WEAVIATE_URL, WEAVIATE_API_KEY
from .config_updater import ConfigUpdater


class OntologyManager:
    def __init__(self) -> None:
        self._client: Optional[weaviate.Client] = None
        self.config_updater = ConfigUpdater()

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

    async def create_and_load_ontology_collection(
        self, collection_name: str, ontology_terms: List[Dict], openai_api_key: str
    ) -> None:
        """Create a Weaviate collection and load ontology terms into it."""
        client = await self.get_weaviate_client()

        existing = client.schema.get().get("classes", [])
        if any(c.get("class") == collection_name for c in existing):
            client.schema.delete_class(collection_name)

        schema = {
            "class": collection_name,
            "vectorizer": "text2vec-openai",
            "properties": [
                {"name": "id", "dataType": ["text"]},
                {"name": "name", "dataType": ["text"]},
                {"name": "definition", "dataType": ["text"]},
            ],
        }
        client.schema.create_class(schema)

        with client.batch.dynamic() as batch:
            for term in ontology_terms:
                batch.add_data_object(
                    data_object={
                        "id": term["id"],
                        "name": term["name"],
                        "definition": term.get("definition", ""),
                    },
                    class_name=collection_name,
                )
        await asyncio.to_thread(client.batch.flush)
