import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import asyncio

from .ontology_manager import OntologyManager
from .ontology_searcher import OntologySearcher
from .llm_matcher import LLMMatcher
from .config_updater import ConfigUpdater
from .config import ADMIN_API_KEY, OPENAI_API_KEY
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
    logger.info("Resolve request for ontology %s", payload.ontology_name)
    collection = ontology_manager.get_current_ontology_version(payload.ontology_name)
    if not collection:
        raise HTTPException(status_code=404, detail="Ontology not configured")
    try:
        candidates = await searcher.search_ontology(payload.passage, collection)
        match = await matcher.select_best_match(payload.passage, candidates)
        if "error" in match:
            return ResolveResponse(error=match["error"])
        best = OntologyTerm(id=match.get("id"), name=match.get("name"))
        return ResolveResponse(
            best_match=best,
            confidence=match.get("confidence"),
            reason=match.get("reason"),
            alternatives=[OntologyTerm(**c) for c in candidates if c.get("id") != best.id],
        )
    except Exception as e:
        logger.exception("Resolution failed")
        raise HTTPException(status_code=500, detail=str(e))


async def _perform_ontology_update(ontology_name: str, source_url: str):
    """Download the ontology from ``source_url`` and load it into Weaviate."""
    logger.info("Starting ontology update for %s from %s", ontology_name, source_url)
    try:
        parsed_go_terms = []

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(source_url) as response:
                    if response.status != 200:
                        logger.error(
                            "Failed to download ontology from %s. Status: %s",
                            source_url,
                            response.status,
                        )
                        raise ValueError(
                            f"Failed to download ontology. Status: {response.status}"
                        )
                    try:
                        data = await response.json()
                    except Exception as exc:
                        logger.exception("Error parsing JSON from %s", source_url)
                        raise exc
            except aiohttp.ClientError as exc:
                logger.exception("Ontology download failed from %s", source_url)
                raise exc

        nodes = []
        if isinstance(data, dict):
            graphs = data.get("graphs", [])
            if graphs and isinstance(graphs, list):
                nodes = graphs[0].get("nodes", [])
        logger.info("Attempting to parse %s nodes from %s", len(nodes), source_url)

        for node in nodes:
            try:
                id_uri = node["id"]
                name = node["lbl"]
            except KeyError:
                logger.warning(
                    "Skipping node due to missing 'id' or 'lbl': %s",
                    node.get("id", "N/A"),
                )
                continue

            term_id = id_uri.split("/")[-1].replace("_", ":")
            definition = (
                node.get("meta", {})
                .get("definition", {})
                .get("val", "")
            )
            parsed_go_terms.append(
                {"id": term_id, "name": name, "definition": definition}
            )

        logger.info(
            "Successfully parsed %s terms from %s", len(parsed_go_terms), source_url
        )

        new_collection = f"{ontology_name}_{int(datetime.utcnow().timestamp())}"
        await ontology_manager.create_and_load_ontology_collection(
            new_collection, parsed_go_terms, OPENAI_API_KEY
        )
        config_updater.update_ontology_version(
            ontology_name, new_collection, source_url
        )
        logger.info("Ontology update completed for %s", ontology_name)
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("Ontology update failed")
        raise exc


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
