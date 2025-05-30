import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import aiofiles
import asyncio
import requests
import openai

from .ontology_manager import OntologyManager
from .ontology_searcher import OntologySearcher
from .llm_matcher import LLMMatcher
from .config_updater import ConfigUpdater, DownloadHistoryManager
from .config import ADMIN_API_KEY, OPENAI_API_KEY, ONTOLOGY_CONFIG, load_ontology_config, EMBEDDINGS_CONFIG, load_embeddings_config
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
embedding_progress_store = {}
# Track background tasks for cancellation
background_tasks_store = {}
# Global store for cancellation flags
embedding_cancellation_flags = {}

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
    """Download the ontology from ``source_url`` and save to disk."""
    
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

        update_progress("completed", 100, f"Download completed! File saved to {filename}")
        add_log(f"Ontology download completed successfully. File: {filename}")
        
        # Store download metadata
        download_info = {
            "ontology_name": ontology_name,
            "source_url": source_url,
            "filename": filename,
            "file_path": file_path,
            "downloaded_at": datetime.utcnow().isoformat(),
            "size_bytes": downloaded_size
        }
        
        # Update the download history so Streamlit can see it
        download_history_manager = DownloadHistoryManager()
        # Store filename with subdirectory path relative to data dir
        filename_with_path = os.path.join("source_ontologies", filename)
        download_history_manager.add_download_record(
            ontology_name=ontology_name,
            filename=filename_with_path,
            size_bytes=downloaded_size,
            timestamp=download_info["downloaded_at"] + "Z"
        )
        # Update status to available since we just downloaded it
        download_history_manager.update_file_status(ontology_name, filename_with_path, "available")
        add_log(f"Updated download history for {ontology_name}")
        
        return download_info
        
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

@app.get("/admin/verify_downloads")
async def verify_downloads(api_key: str = Depends(verify_api_key)):
    """Verify all download records and update their status."""
    download_manager = DownloadHistoryManager()
    verification_results = download_manager.verify_all_downloads()
    
    return {
        "verified_at": datetime.utcnow().isoformat() + "Z",
        "results": verification_results
    }

@app.get("/admin/download_status/{ontology_name}")
async def get_download_status(ontology_name: str, api_key: str = Depends(verify_api_key)):
    """Get the download status for a specific ontology."""
    download_manager = DownloadHistoryManager()
    
    # Get the latest available download
    latest_download = download_manager.get_latest_available_download(ontology_name)
    
    # Get full history
    all_history = download_manager.get_download_history()
    ontology_history = all_history.get(ontology_name, [])
    
    # Determine overall status
    if not ontology_history:
        status = "not_downloaded"
    elif latest_download:
        status = "ready_for_embedding"
    else:
        # Check if any files exist
        any_exist = any(
            download_manager.verify_file_exists(record.get("filename", ""))
            for record in ontology_history
        )
        status = "file_missing" if not any_exist else "unknown"
    
    return {
        "ontology_name": ontology_name,
        "status": status,
        "latest_available": latest_download,
        "total_downloads": len(ontology_history),
        "history": ontology_history
    }

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
        global ONTOLOGY_CONFIG, EMBEDDINGS_CONFIG
        ONTOLOGY_CONFIG = load_ontology_config()
        EMBEDDINGS_CONFIG = load_embeddings_config()
        
        logger.info("Configuration reloaded successfully")
        return {
            "status": "success",
            "message": "Configuration reloaded successfully",
            "ontologies": list(ONTOLOGY_CONFIG.get("ontologies", {}).keys()),
            "embedding_model": EMBEDDINGS_CONFIG.get("model", {}).get("name")
        }
    except Exception as e:
        logger.exception("Failed to reload configuration")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")


async def _generate_embeddings_only(ontology_name: str, collection_name: str = None):
    """Generate embeddings for an existing ontology without downloading."""
    
    # Initialize progress tracking and cancellation flag
    progress_key = f"{ontology_name}_embeddings"
    cancellation_key = f"{ontology_name}_cancel"
    
    embedding_progress_store[progress_key] = {
        "status": "starting",
        "progress_percentage": 0,
        "recent_logs": [],
        "started_at": time.time(),
        "ontology_name": ontology_name,
        "collection_name": collection_name,
        "embedding_stats": {}
    }
    embedding_cancellation_flags[cancellation_key] = False
    
    def add_log(message: str, level: str = "INFO"):
        """Add a log entry to progress tracking."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        if progress_key in embedding_progress_store:
            embedding_progress_store[progress_key]["recent_logs"].append(log_entry)
            # Keep only last 10 logs
            embedding_progress_store[progress_key]["recent_logs"] = embedding_progress_store[progress_key]["recent_logs"][-10:]
        logger.info("Embeddings %s: %s", ontology_name, message)

    def update_progress(status: str, percentage: int, message: str = "", **kwargs):
        """Update progress status."""
        if progress_key in embedding_progress_store:
            embedding_progress_store[progress_key].update({
                "status": status,
                "progress_percentage": percentage
            })
            if kwargs:
                embedding_progress_store[progress_key].update(kwargs)
            if message:
                add_log(message)

    try:
        add_log(f"Starting embedding generation for {ontology_name}")
        update_progress("starting", 5, "Initializing embedding generation...")

        # Get the persistent data directory from environment
        data_dir = os.environ.get("ONTOLOGY_DATA_DIR", "/app/data")
        source_ontologies_dir = os.path.join(data_dir, "source_ontologies")
        filename = f"{ontology_name}.json"
        file_path = os.path.join(source_ontologies_dir, filename)
        
        # Check if source file exists
        if not os.path.exists(file_path):
            error_msg = f"Source file not found: {file_path}"
            add_log(error_msg, "ERROR")
            update_progress("failed", 0, error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
            
        update_progress("loading", 10, f"Loading ontology data from {filename}")
        
        # Load the ontology data
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            add_log("Successfully loaded JSON data")
        except Exception as exc:
            error_msg = f"Error loading JSON: {str(exc)}"
            add_log(error_msg, "ERROR")
            update_progress("failed", 0, error_msg)
            raise exc

        update_progress("parsing", 20, "Parsing ontology terms...")

        # Get ontology-specific configuration
        ontology_config = get_ontology_config(ontology_name)
        
        # Use enhanced GO parser for comprehensive data extraction
        id_format = ontology_config.get("id_format", {"prefix_replacement": {"_": ":"}})
        parsed_terms = parse_go_json_enhanced(data, id_format)

        add_log(f"Successfully parsed {len(parsed_terms)} terms")
        update_progress("embedding", 30, f"Creating embeddings for {len(parsed_terms)} terms...")

        # Generate new collection name if not provided
        if not collection_name:
            collection_name = f"{ontology_name}_{int(datetime.utcnow().timestamp())}"
            
        # Create embedding progress callback
        def embedding_progress_callback(status: str, percentage: int, message: str, extra_data: dict):
            """Update embedding progress."""
            if progress_key in embedding_progress_store:
                embedding_progress_store[progress_key]["embedding_stats"] = extra_data
                # Map embedding progress to overall progress (30-95%)
                overall_percentage = 30 + int(percentage * 0.65)
                update_progress(
                    status,
                    overall_percentage,
                    message,
                    embedding_stats=extra_data
                )
        
        # Create cancellation check callback
        def cancellation_check() -> bool:
            """Check if the operation should be cancelled."""
            return embedding_cancellation_flags.get(cancellation_key, False)
        
        await ontology_manager.create_and_load_ontology_collection(
            collection_name, parsed_terms, OPENAI_API_KEY, embedding_progress_callback, cancellation_check
        )
        
        update_progress("finalizing", 96, "Updating configuration...")
        
        # Update configuration to use new collection
        config_updater.update_ontology_version(
            ontology_name, collection_name, "embeddings_only"
        )
        
        # Check if cancelled
        if embedding_cancellation_flags.get(cancellation_key, False):
            add_log("Embedding generation was cancelled", "WARNING")
            update_progress("cancelled", embedding_progress_store[progress_key].get("progress_percentage", 0), "Operation cancelled")
        else:
            add_log(f"Embedding generation completed successfully. Collection: {collection_name}")
            update_progress("completed", 100, "Embeddings generated successfully!")
        
        return collection_name
        
    except Exception as exc:
        error_msg = f"Embedding generation failed: {str(exc)}"
        add_log(error_msg, "ERROR")
        update_progress("failed", embedding_progress_store[progress_key].get("progress_percentage", 0), error_msg)
        logger.exception("Embedding generation failed")
        raise exc
    finally:
        # Clean up cancellation flag
        if cancellation_key in embedding_cancellation_flags:
            del embedding_cancellation_flags[cancellation_key]


@app.post("/admin/generate_embeddings")
async def generate_embeddings(
    request: Dict[str, str],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate embeddings for an existing ontology file."""
    ontology_name = request.get("ontology_name")
    collection_name = request.get("collection_name", None)
    
    if not ontology_name:
        raise HTTPException(status_code=400, detail="ontology_name is required")
        
    background_tasks.add_task(
        _generate_embeddings_only, ontology_name, collection_name
    )
    return {"status": "embedding generation started"}


@app.get("/admin/embedding_progress/{ontology_name}")
async def get_embedding_progress(ontology_name: str, api_key: str = Depends(verify_api_key)):
    """Get progress information for embedding generation."""
    progress_key = f"{ontology_name}_embeddings"
    if progress_key not in embedding_progress_store:
        raise HTTPException(status_code=404, detail="No active embedding generation found for this ontology")
    
    progress_data = embedding_progress_store[progress_key].copy()
    
    # Calculate elapsed time
    elapsed_time = time.time() - progress_data.get("started_at", time.time())
    progress_data["elapsed_seconds"] = int(elapsed_time)
    
    # Add embedding-specific stats
    embedding_stats = progress_data.get("embedding_stats", {})
    if embedding_stats:
        progress_data["terms_processed"] = embedding_stats.get("processed_terms", 0)
        progress_data["terms_failed"] = embedding_stats.get("failed_terms", 0)
        progress_data["batches_completed"] = embedding_stats.get("batches_completed", 0)
        progress_data["total_batches"] = embedding_stats.get("total_batches", 0)
        
        # Calculate rate
        if elapsed_time > 0 and embedding_stats.get("processed_terms", 0) > 0:
            progress_data["terms_per_second"] = round(
                embedding_stats["processed_terms"] / elapsed_time, 2
            )
    
    return progress_data


@app.delete("/admin/embedding_progress/{ontology_name}")
async def cancel_embedding_generation(ontology_name: str, api_key: str = Depends(verify_api_key)):
    """Cancel an ongoing embedding generation."""
    progress_key = f"{ontology_name}_embeddings"
    cancellation_key = f"{ontology_name}_cancel"
    
    if progress_key not in embedding_progress_store:
        raise HTTPException(status_code=404, detail="No active embedding generation found for this ontology")
    
    # Set cancellation flag
    embedding_cancellation_flags[cancellation_key] = True
    
    # Mark as cancelled in progress store
    if progress_key in embedding_progress_store:
        embedding_progress_store[progress_key]["status"] = "cancelling"
        embedding_progress_store[progress_key]["recent_logs"].append({
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "message": "Cancellation requested by user",
            "level": "WARNING"
        })
    
    return {"status": "Cancellation requested. The operation will stop at the next safe point."}


@app.get("/admin/embeddings_config")
async def get_embeddings_config(api_key: str = Depends(verify_api_key)):
    """Get current embeddings configuration."""
    return EMBEDDINGS_CONFIG


@app.post("/admin/test_embeddings_config")
async def test_embeddings_config(api_key: str = Depends(verify_api_key)):
    """Test embeddings configuration by embedding a sample text."""
    try:
        # Use a simple test string
        test_text = "This is a test embedding for configuration validation."
        
        # Get model configuration
        model_name = EMBEDDINGS_CONFIG.get("model", {}).get("name", "text-ada-002")
        
        # Test OpenAI API
        client = openai.Client(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            model=model_name,
            input=test_text
        )
        
        # Extract embedding info
        embedding = response.data[0].embedding
        dimensions = len(embedding)
        
        return {
            "success": True,
            "model": model_name,
            "dimensions": dimensions,
            "test_text": test_text,
            "embedding_preview": embedding[:5] if embedding else [],  # First 5 dimensions
            "message": f"Successfully created {dimensions}-dimensional embedding with {model_name}"
        }
        
    except openai.AuthenticationError:
        return {
            "success": False,
            "error": "authentication_error",
            "message": "Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
        }
    except openai.NotFoundError:
        return {
            "success": False,
            "error": "model_not_found",
            "message": f"Model '{model_name}' not found. Please check the model name in embeddings configuration."
        }
    except openai.RateLimitError:
        return {
            "success": False,
            "error": "rate_limit",
            "message": "OpenAI API rate limit exceeded. Please try again later."
        }
    except Exception as e:
        return {
            "success": False,
            "error": "unknown_error",
            "message": f"Failed to test embeddings: {str(e)}"
        }
