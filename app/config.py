import os
import yaml
from typing import Dict, Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))

RUNTIME_CONFIG_PATH = os.getenv(
    "RUNTIME_CONFIG_PATH", os.path.join(os.getcwd(), "ontology_versions.json")
)

ONTOLOGY_CONFIG_PATH = os.getenv(
    "ONTOLOGY_CONFIG_PATH", os.path.join(os.getcwd(), "ontology_config.yaml")
)

def load_ontology_config() -> Dict[str, Any]:
    """Load ontology configuration from YAML file."""
    try:
        with open(ONTOLOGY_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "ontologies": {
                "GO": {"name": "Gene Ontology", "enabled": True},
                "DOID": {"name": "Disease Ontology", "enabled": True}
            },
            "settings": {"default_k": 5}
        }

# Load ontology configuration
ONTOLOGY_CONFIG = load_ontology_config()
