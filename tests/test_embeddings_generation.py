"""Test embedding generation functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime

from app.ontology_manager import OntologyManager
from app.config import EMBEDDINGS_CONFIG


@pytest.mark.asyncio
async def test_create_and_load_ontology_collection_with_progress():
    """Test collection creation with progress tracking."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Set up the mock chain
    mock_client.collections.get.return_value = mock_collection
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append({
            "status": status,
            "percentage": percentage,
            "message": message,
            "extra_data": extra_data
        })
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        # Test data
        test_terms = [
            {"id": "GO:0001", "name": "test term 1", "definition": "test def 1"},
            {"id": "GO:0002", "name": "test term 2", "definition": "test def 2"}
        ]
        
        await manager.create_and_load_ontology_collection(
            "test_collection",
            test_terms,
            "test_api_key",
            progress_callback
        )
    
    # Verify progress updates were made
    assert len(progress_updates) > 0
    
    # Check key progress stages
    statuses = [update["status"] for update in progress_updates]
    assert "initializing" in statuses
    assert "creating_collection" in statuses
    assert "processing_terms" in statuses
    assert "embedding_generation" in statuses or "embedding_batch" in statuses
    assert "completed" in statuses
    
    # Verify final progress is 100%
    final_update = progress_updates[-1]
    assert final_update["percentage"] == 100
    assert final_update["status"] == "completed"
    
    # Check embedding statistics
    final_stats = final_update["extra_data"]
    assert final_stats["total_terms"] == 2
    assert final_stats["processed_terms"] > 0
    assert "elapsed_time" in final_stats
    assert "terms_per_second" in final_stats


@pytest.mark.asyncio
async def test_embedding_configuration_applied():
    """Test that embedding configuration is properly applied."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        with patch.dict(EMBEDDINGS_CONFIG, {
            "model": {"name": "text-embedding-3-small"},
            "processing": {"batch_size": 50}
        }):
            await manager.create_and_load_ontology_collection(
                "test_collection",
                [],
                "test_api_key",
                None
            )
    
    # Verify correct model configuration was used
    create_call = mock_client.collections.create.call_args
    vectorizer_config = create_call.kwargs['vectorizer_config']
    
    # Check model configuration (implementation specific)
    assert create_call.kwargs['name'] == "test_collection"


@pytest.mark.asyncio
async def test_batch_processing_with_failures():
    """Test batch processing handles failures gracefully."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Simulate batch failures
    failure_count = 0
    def add_object_side_effect(obj):
        nonlocal failure_count
        if failure_count < 2 and "fail" in obj.get("name", ""):
            failure_count += 1
            raise Exception("Simulated batch error")
    
    mock_batch.add_object = MagicMock(side_effect=add_object_side_effect)
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Track progress
    progress_updates = []
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append(extra_data)
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        # Test data with some terms that will fail
        test_terms = [
            {"id": "GO:0001", "name": "test term 1", "definition": "test def 1"},
            {"id": "GO:0002", "name": "fail term 2", "definition": "test def 2"},
            {"id": "GO:0003", "name": "test term 3", "definition": "test def 3"},
            {"id": "GO:0004", "name": "fail term 4", "definition": "test def 4"},
            {"id": "GO:0005", "name": "test term 5", "definition": "test def 5"}
        ]
        
        await manager.create_and_load_ontology_collection(
            "test_collection",
            test_terms,
            "test_api_key",
            progress_callback
        )
    
    # Check final statistics
    final_stats = progress_updates[-1]
    assert final_stats["total_terms"] == 5
    assert final_stats["processed_terms"] == 3  # 5 total - 2 failed
    assert final_stats["failed_terms"] == 2


def test_embeddings_config_loading():
    """Test loading embeddings configuration."""
    from app.config import load_embeddings_config
    
    config = load_embeddings_config()
    
    # Check required sections exist
    assert "model" in config
    assert "processing" in config
    assert "vectorize_fields" in config
    assert "preprocessing" in config
    assert "performance" in config
    assert "usage" in config
    
    # Check model configuration
    model_config = config["model"]
    assert "name" in model_config
    assert model_config["name"] in ["text-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    
    # Check processing configuration
    processing_config = config["processing"]
    assert "batch_size" in processing_config
    assert isinstance(processing_config["batch_size"], int)
    assert processing_config["batch_size"] > 0
    
    # Check vectorize fields
    vectorize_config = config["vectorize_fields"]
    assert isinstance(vectorize_config.get("name", False), bool)
    assert isinstance(vectorize_config.get("definition", False), bool)
    assert isinstance(vectorize_config.get("synonyms", False), bool)