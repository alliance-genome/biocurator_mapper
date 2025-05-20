import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))

RUNTIME_CONFIG_PATH = os.getenv(
    "RUNTIME_CONFIG_PATH", os.path.join(os.getcwd(), "ontology_versions.json")
)
