import json
import logging
import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import aiofiles
import asyncio
import requests

from .ontology_manager import OntologyManager
from .ontology_searcher import OntologySearcher
from .llm_matcher import LLMMatcher
from .config_updater import ConfigUpdater
from .config import ADMIN_API_KEY, OPENAI_API_KEY, ONTOLOGY_CONFIG, load_ontology_config
from .go_parser import parse_go_json_enhanced
from .ontology_version_manager import OntologyVersionManager
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
version_manager = OntologyVersionManager()

# Progress tracking store (in-memory for now)
update_progress_store = {}
# Track background tasks for cancellation
background_tasks_store = {}

def verify_api_key(x_api_key: str = Header(...)):
    if ADMIN_API_KEY and x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "biocurator-mapper-api"
    }


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


def get_ontology_config(ontology_name: str) -> dict:
    """Get configuration for a specific ontology."""
    return ONTOLOGY_CONFIG.get("ontologies", {}).get(ontology_name, {})

def get_nested_value(data: dict, path: list, default=""):
    """Get nested value from dict using path list."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

async def _perform_ontology_update(ontology_name: str, source_url: str):
    """Download the ontology from ``source_url``, save to disk, and load it into Weaviate."""
    
    # Initialize progress tracking
    progress_key = ontology_name
    update_progress_store[progress_key] = {
        "status": "starting",
        "progress_percentage": 0,
        "recent_logs": [],
        "started_at": time.time(),
        "ontology_name": ontology_name,
        "source_url": source_url,
        "download_percentage": 0,
        "download_bytes": 0,
        "download_total_bytes": 0
    }
    
    def add_log(message: str, level: str = "INFO"):
        """Add a log entry to progress tracking."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        if progress_key in update_progress_store:
            update_progress_store[progress_key]["recent_logs"].append(log_entry)
            # Keep only last 10 logs
            update_progress_store[progress_key]["recent_logs"] = update_progress_store[progress_key]["recent_logs"][-10:]
        logger.info("Update %s: %s", ontology_name, message)

    def update_progress(status: str, percentage: int, message: str = "", 
                       download_percentage: int = None, download_bytes: int = None, 
                       download_total_bytes: int = None):
        """Update progress status."""
        if progress_key in update_progress_store:
            update_progress_store[progress_key].update({
                "status": status,
                "progress_percentage": percentage
            })
            if download_percentage is not None:
                update_progress_store[progress_key]["download_percentage"] = download_percentage
            if download_bytes is not None:
                update_progress_store[progress_key]["download_bytes"] = download_bytes
            if download_total_bytes is not None:
                update_progress_store[progress_key]["download_total_bytes"] = download_total_bytes
            if message:
                add_log(message)

    try:
        add_log(f"Starting ontology update from {source_url}")
        update_progress("starting", 5, "Initializing download...")

        # Get the persistent data directory from environment
        data_dir = os.environ.get("ONTOLOGY_DATA_DIR", "/app/data")
        
        # Create source_ontologies subdirectory
        source_ontologies_dir = os.path.join(data_dir, "source_ontologies")
        os.makedirs(source_ontologies_dir, exist_ok=True)
        
        # Standardized filename for this ontology
        filename = f"{ontology_name}.json"
        file_path = os.path.join(source_ontologies_dir, filename)
        
        update_progress("downloading", 10, f"Downloading to {filename}")

        # Download function to run in thread
        def download_file_with_progress():
            """Download file with progress tracking using requests."""
            try:
                with requests.get(source_url, stream=True) as response:
                    response.raise_for_status()
                    
                    # Get file size for progress tracking
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    # Update progress with total size
                    update_progress("downloading", 10, f"Starting download ({total_size // 1024 // 1024} MB)",
                                  download_percentage=0, download_bytes=0, download_total_bytes=total_size)
                    
                    # Stream download to file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            # Check for cancellation
                            if (progress_key in update_progress_store and 
                                update_progress_store[progress_key].get("status") == "cancelled"):
                                logger.warning("Download cancelled by user")
                                return None
                                
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                # Update download progress
                                if total_size > 0:
                                    download_pct = int((downloaded_size / total_size) * 100)
                                    overall_pct = 10 + int((downloaded_size / total_size) * 30)  # 30% for download
                                    update_progress("downloading", overall_pct, 
                                                  f"Downloaded {downloaded_size // 1024 // 1024} MB of {total_size // 1024 // 1024} MB",
                                                  download_percentage=download_pct,
                                                  download_bytes=downloaded_size,
                                                  download_total_bytes=total_size)
                    
                    return downloaded_size
                    
            except requests.RequestException as exc:
                raise exc
        
        # Run download in thread to not block async event loop
        try:
            downloaded_size = await asyncio.to_thread(download_file_with_progress)
            
            if downloaded_size is None:
                add_log("Download cancelled by user", "WARNING")
                update_progress("cancelled", 0, "Download cancelled")
                return None
                
            add_log(f"Downloaded {downloaded_size // 1024 // 1024} MB to {filename}")
            
        except Exception as exc:
            error_msg = f"Download failed: {str(exc)}"
            add_log(error_msg, "ERROR")
            update_progress("failed", 0, error_msg)
            raise exc

        update_progress("parsing", 45, "Parsing JSON file...")
        
        # Load and parse the downloaded file from persistent location
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            add_log("Successfully parsed JSON data")
        except Exception as exc:
            error_msg = f"Error parsing JSON: {str(exc)}"
            add_log(error_msg, "ERROR")
            update_progress("failed", 0, error_msg)
            raise exc

        update_progress("version_check", 50, "Checking version information...")

        # Get ontology-specific configuration
        ontology_config = get_ontology_config(ontology_name)
        
        # Check if version comparison indicates update is needed
        needs_update, stored_info, new_version_info = version_manager.compare_versions(
            ontology_name, data
        )
        
        if not needs_update and stored_info:
            add_log(f"Ontology is up to date (version: {stored_info.get('version_info', {}).get('version_date', 'unknown')})")
            update_progress("completed", 100, "Ontology already up to date")
            
            # Update config to point to existing collection
            config_updater.update_ontology_version(
                ontology_name, stored_info['collection_name'], source_url
            )
            
            return stored_info['collection_name']

        add_log(f"New version detected: {new_version_info.get('version_date', 'unknown')}")
        update_progress("processing", 60, "Processing ontology terms...")

        # Use enhanced GO parser for comprehensive data extraction
        id_format = ontology_config.get("id_format", {"prefix_replacement": {"_": ":"}})
        parsed_terms = parse_go_json_enhanced(data, id_format)

        add_log(f"Successfully parsed {len(parsed_terms)} terms")
        update_progress("embedding", 70, f"Creating embeddings for {len(parsed_terms)} terms...")

        new_collection = f"{ontology_name}_{int(datetime.utcnow().timestamp())}"
        await ontology_manager.create_and_load_ontology_collection(
            new_collection, parsed_terms, OPENAI_API_KEY
        )
        
        update_progress("finalizing", 90, "Storing version metadata...")
        
        # Store version metadata for future comparisons
        version_manager.store_version_info(ontology_name, new_version_info, new_collection, source_url)
        
        config_updater.update_ontology_version(
            ontology_name, new_collection, source_url
        )
        
        add_log(f"Ontology update completed successfully. Collection: {new_collection}")
        update_progress("completed", 100, "Update completed successfully!")
        
        return new_collection
        
    except Exception as exc:
        error_msg = f"Ontology update failed: {str(exc)}"
        add_log(error_msg, "ERROR")
        update_progress("failed", 0, error_msg)
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


@app.get("/admin/update_progress/{ontology_name}")
async def get_update_progress(ontology_name: str, api_key: str = Depends(verify_api_key)):
    """Get progress information for an ontology update."""
    if ontology_name not in update_progress_store:
        raise HTTPException(status_code=404, detail="No active update found for this ontology")
    
    progress_data = update_progress_store[ontology_name].copy()
    
    # Calculate elapsed time
    elapsed_time = time.time() - progress_data.get("started_at", time.time())
    progress_data["elapsed_seconds"] = int(elapsed_time)
    
    return progress_data

@app.delete("/admin/update_progress/{ontology_name}")
async def cancel_update(ontology_name: str, api_key: str = Depends(verify_api_key)):
    """Cancel an ongoing ontology update."""
    if ontology_name not in update_progress_store:
        raise HTTPException(status_code=404, detail="No active update found for this ontology")
    
    # Mark as cancelled in progress store
    if ontology_name in update_progress_store:
        update_progress_store[ontology_name]["status"] = "cancelled"
        update_progress_store[ontology_name]["recent_logs"].append({
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "message": "Update cancelled by user",
            "level": "WARNING"
        })
    
    return {"status": "Update marked for cancellation"}

@app.get("/admin/ontology_status")
async def ontology_status(api_key: str = Depends(verify_api_key)):
    return config_updater.get_all_ontology_configs()

@app.get("/ontology_config")
async def get_ontology_config_endpoint():
    """Get ontology configuration information."""
    return {
        "ontologies": {
            name: {
                "name": config.get("name", name),
                "description": config.get("description", ""),
                "enabled": config.get("enabled", True),
                "default_source_url": config.get("default_source_url", "")
            }
            for name, config in ONTOLOGY_CONFIG.get("ontologies", {}).items()
            if config.get("enabled", True)
        }
    }


@app.get("/admin/weaviate_health")
async def weaviate_health_check(api_key: str = Depends(verify_api_key)):
    """Check Weaviate health status."""
    try:
        # Check if Weaviate client is healthy
        is_healthy = await ontology_manager.check_weaviate_health()
        return {
            "healthy": is_healthy,
            "details": "Weaviate is connected and operational" if is_healthy else "Cannot connect to Weaviate"
        }
    except Exception as e:
        return {
            "healthy": False,
            "details": f"Error checking Weaviate health: {str(e)}"
        }


@app.get("/admin/openai_health")
async def openai_health_check(api_key: str = Depends(verify_api_key)):
    """Check OpenAI API health status."""
    try:
        if not OPENAI_API_KEY:
            return {
                "healthy": False,
                "details": "OpenAI API key not configured"
            }
        
        # Do a simple test with the OpenAI API
        is_healthy = await matcher.check_openai_health()
        return {
            "healthy": is_healthy,
            "details": "OpenAI API is accessible" if is_healthy else "Cannot connect to OpenAI API"
        }
    except Exception as e:
        return {
            "healthy": False,
            "details": f"Error checking OpenAI health: {str(e)}"
        }


@app.post("/admin/reload_config")
async def reload_config(api_key: str = Depends(verify_api_key)):
    """Reload ontology configuration from file."""
    try:
        # Reload the configuration
        global ONTOLOGY_CONFIG
        ONTOLOGY_CONFIG = load_ontology_config()
        
        logger.info("Configuration reloaded successfully")
        return {
            "status": "success",
            "message": "Configuration reloaded successfully",
            "ontologies": list(ONTOLOGY_CONFIG.get("ontologies", {}).keys())
        }
    except Exception as e:
        logger.exception("Failed to reload configuration")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")
