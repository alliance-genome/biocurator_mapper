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

EMBEDDINGS_CONFIG_PATH = os.getenv(
    "EMBEDDINGS_CONFIG_PATH", os.path.join(os.getcwd(), "embeddings_config.yaml")
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

def load_embeddings_config() -> Dict[str, Any]:
    """Load embeddings configuration from YAML file."""
    try:
        with open(EMBEDDINGS_CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "model": {
                "name": "text-ada-002",
                "dimensions": 1536
            },
            "processing": {
                "batch_size": 100,
                "parallel_processing": True,
                "retry_failed": True,
                "max_retries": 3
            },
            "vectorize_fields": {
                "name": True,
                "definition": True,
                "synonyms": True
            },
            "preprocessing": {
                "lowercase": False,
                "remove_punctuation": False,
                "combine_fields_separator": " | "
            },
            "performance": {
                "request_timeout": 30,
                "rate_limit_delay": 0.1
            },
            "usage": {
                "track_tokens": True,
                "log_requests": False
            }
        }

# Load ontology configuration
ONTOLOGY_CONFIG = load_ontology_config()

# Load embeddings configuration
EMBEDDINGS_CONFIG = load_embeddings_config()


class Config:
    """Configuration management class."""
    
    def __init__(self):
        self.embeddings_config_path = EMBEDDINGS_CONFIG_PATH
        
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return load_embeddings_config()
