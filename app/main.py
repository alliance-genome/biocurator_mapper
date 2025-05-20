import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware

from .ontology_manager import OntologyManager
from .ontology_searcher import OntologySearcher
from .llm_matcher import LLMMatcher
from .config_updater import ConfigUpdater
from .config import ADMIN_API_KEY
from .models import (
    ResolveRequest,
    ResolveResponse,
    OntologyUpdateRequest,
    OntologyTerm,
)

app = FastAPI(title="Biocurator Mapper")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

ontology_manager = OntologyManager()
searcher = OntologySearcher(ontology_manager)
matcher = LLMMatcher()
config_updater = ConfigUpdater()


def verify_api_key(x_api_key: str = Header(...)):
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/resolve_biocurated_data", response_model=ResolveResponse)
async def resolve_biocurated_data(payload: ResolveRequest):
    collection = ontology_manager.get_current_ontology_version(payload.ontology_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Ontology not configured")
    try:
        candidates = searcher.search_ontology(payload.passage, collection)
        match = matcher.select_best_match(payload.passage, candidates)
        if "error" in match:
            return ResolveResponse(error=match["error"])
        best = OntologyTerm(id=match.get("id"), name=match.get("name"))
        return ResolveResponse(
            best_match=best,
            confidence=match.get("confidence"),
            reason=match.get("reason"),
            alternatives=[OntologyTerm(**c) for c in candidates],
        )
    except Exception as e:
        logger.exception("Resolution failed")
        raise HTTPException(status_code=500, detail=str(e))


def _perform_ontology_update(ontology_name: str, source_url: str):
    # Placeholder for actual ontology ingestion process
    # Download ontology, parse, embed, and load into new Weaviate collection
    new_collection = f"{ontology_name}_TODO_TIMESTAMP"
    config_updater.update_ontology_version(ontology_name, new_collection, source_url)


@app.post("/admin/update_ontology")
async def update_ontology(
    request: OntologyUpdateRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    background_tasks.add_task(
        _perform_ontology_update, request.ontology_name, request.source_url
    )
    return {"status": "update started"}


@app.get("/admin/ontology_status")
async def ontology_status(api_key: str = Depends(verify_api_key)):
    return config_updater.get_all_ontology_configs()
