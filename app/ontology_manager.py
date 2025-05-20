from typing import Optional

import weaviate

from .config import WEAVIATE_URL, WEAVIATE_API_KEY
from .config_updater import ConfigUpdater


class OntologyManager:
    def __init__(self) -> None:
        self._client: Optional[weaviate.Client] = None
        self.config_updater = ConfigUpdater()

    def _init_client(self) -> weaviate.Client:
        auth = None
        if WEAVIATE_API_KEY:
            auth = weaviate.AuthApiKey(WEAVIATE_API_KEY)
        client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth)
        return client

    def get_weaviate_client(self) -> weaviate.Client:
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def get_current_ontology_version(self, ontology_name: str) -> Optional[str]:
        return self.config_updater.get_current_ontology_version(ontology_name)
