"""Test embedding generation functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio
from datetime import datetime
import time

from app.ontology_manager import OntologyManager
from app.config import EMBEDDINGS_CONFIG
from app.main import _generate_embeddings_only, embedding_progress_store, embedding_cancellation_flags
import app.main


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
            "processing": {"batch_size": 50},
            "vectorize_fields": {"name": True, "definition": False, "synonyms": True}
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


@pytest.mark.asyncio
async def test_openai_rate_limit_handling():
    """Test handling of OpenAI rate limit errors."""
    from openai import RateLimitError
    
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Simulate rate limit error on first attempt, success on retry
    call_count = 0
    def batch_context_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RateLimitError("Rate limit exceeded", response=None, body=None)
        return mock_batch
    
    mock_collection.batch.dynamic.side_effect = batch_context_side_effect
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Track progress for retry detection
    progress_updates = []
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append({
            "status": status,
            "message": message,
            "extra_data": extra_data
        })
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        with patch.dict(EMBEDDINGS_CONFIG, {
            "processing": {"retry_failed": True, "max_retries": 3}
        }):
            test_terms = [{"id": "DO:0001", "name": "disease 1", "definition": "def 1"}]
            
            await manager.create_and_load_ontology_collection(
                "test_collection",
                test_terms,
                "test_api_key",
                progress_callback
            )
    
    # Verify retry occurred
    statuses = [update["status"] for update in progress_updates]
    assert "rate_limited" in statuses or "retrying_batch" in statuses
    
    # Verify retry count
    final_stats = progress_updates[-1]["extra_data"]
    assert final_stats.get("retry_count", 0) >= 1


@pytest.mark.asyncio
async def test_cancellation_during_embedding():
    """Test cancellation mechanism during embedding generation."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Track when cancellation was checked
    cancellation_checks = []
    cancelled = False
    
    def cancellation_check():
        cancellation_checks.append(time.time())
        # Cancel after a few checks
        if len(cancellation_checks) > 2:
            return True
        return False
    
    # Track progress
    progress_updates = []
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append({
            "status": status,
            "message": message
        })
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        # Large dataset to ensure cancellation happens during processing
        test_terms = [
            {"id": f"DO:{i:04d}", "name": f"disease {i}", "definition": f"def {i}"}
            for i in range(1000)
        ]
        
        await manager.create_and_load_ontology_collection(
            "test_collection",
            test_terms,
            "test_api_key",
            progress_callback,
            cancellation_check
        )
    
    # Verify cancellation was checked
    assert len(cancellation_checks) > 0
    
    # Verify operation was cancelled
    final_status = progress_updates[-1]["status"]
    assert final_status == "cancelled"
    assert "cancelled" in progress_updates[-1]["message"].lower()


@pytest.mark.asyncio
async def test_embeddings_config_loading():
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


@pytest.mark.asyncio
async def test_generate_embeddings_only_task():
    """Test the _generate_embeddings_only background task."""
    
    # Mock dependencies
    mock_ontology_manager = MagicMock()
    mock_ontology_manager.create_and_load_ontology_collection = AsyncMock()
    
    with patch('app.main.OntologyManager', return_value=mock_ontology_manager):
        with patch('app.main.parse_go_json_enhanced') as mock_parser:
            with patch('aiofiles.open', create=True) as mock_aopen:
                with patch('os.path.exists', return_value=True):
                    # Setup mock file reading
                    mock_file = AsyncMock()
                    mock_file.read = AsyncMock(return_value='{"terms": []}')
                    mock_aopen.return_value.__aenter__.return_value = mock_file
                    
                    # Setup mock parser
                    mock_parser.return_value = [
                        {"id": "DOID:0001", "name": "disease 1", "definition": "def 1"}
                    ]
                    
                    # Clear any existing progress
                    embedding_progress_store.clear()
                    embedding_cancellation_flags.clear()
                    
                    # Run the task
                    await _generate_embeddings_only("DOID", "test_collection")
                    
                    # Verify ontology manager was called
                    assert mock_ontology_manager.create_and_load_ontology_collection.called
                    call_args = mock_ontology_manager.create_and_load_ontology_collection.call_args
                    assert call_args[0][0] == "test_collection"
                    assert len(call_args[0][1]) == 1  # One parsed term
                    
                    # Verify progress was tracked
                    progress_key = "DOID_embeddings"
                    assert progress_key in embedding_progress_store
                    assert embedding_progress_store[progress_key]["status"] == "completed"
                    assert embedding_progress_store[progress_key]["progress_percentage"] == 100


@pytest.mark.asyncio
async def test_generate_embeddings_file_not_found():
    """Test handling when source ontology file is not found."""
    
    with patch('os.path.exists', return_value=False):
        # Clear any existing progress
        embedding_progress_store.clear()
        
        # Run the task and expect HTTPException
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await _generate_embeddings_only("DOID", "test_collection")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
        
        # Verify progress shows failure
        progress_key = "DOID_embeddings"
        assert progress_key in embedding_progress_store
        assert embedding_progress_store[progress_key]["status"] == "failed"


@pytest.mark.asyncio
async def test_searchable_text_configuration():
    """Test that searchable text is built according to configuration."""
    manager = OntologyManager()
    
    # Test with different configurations
    test_cases = [
        {
            "config": {
                "vectorize_fields": {"name": True, "definition": True, "synonyms": False},
                "preprocessing": {"lowercase": False, "remove_punctuation": False, "combine_fields_separator": " | "}
            },
            "term_data": {
                "name": "Disease Name",
                "definition": "A disease definition.",
                "all_synonyms": ["Synonym1", "Synonym2"]
            },
            "expected": "Disease Name | A disease definition."
        },
        {
            "config": {
                "vectorize_fields": {"name": True, "definition": False, "synonyms": True},
                "preprocessing": {"lowercase": True, "remove_punctuation": False, "combine_fields_separator": " | "}
            },
            "term_data": {
                "name": "Disease Name",
                "definition": "A disease definition.",
                "all_synonyms": ["Synonym1", "Synonym2"]
            },
            "expected": "disease name | synonym1 | synonym2"
        },
        {
            "config": {
                "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
                "preprocessing": {"lowercase": True, "remove_punctuation": True, "combine_fields_separator": " "}
            },
            "term_data": {
                "name": "Disease-Name!",
                "definition": "A disease definition.",
                "all_synonyms": ["Synonym-1", "Synonym-2"]
            },
            "expected": "diseasename a disease definition synonym1 synonym2"
        }
    ]
    
    for test_case in test_cases:
        # Mock the config
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', test_case["config"]):
            result = manager._build_searchable_text(test_case["term_data"])
            assert result == test_case["expected"], f"Expected '{test_case['expected']}' but got '{result}'"


@pytest.mark.asyncio
async def test_retry_logic_with_rate_limit():
    """Test retry logic when encountering rate limit errors."""
    from openai import RateLimitError
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Track retry attempts
    add_object_call_count = 0
    
    def mock_add_object(obj):
        nonlocal add_object_call_count
        add_object_call_count += 1
        if add_object_call_count <= 2:
            # Fail first two attempts with rate limit
            raise RateLimitError("Rate limit exceeded", response=None, body=None)
        # Success on third attempt
        return True
    
    mock_batch.add_object = mock_add_object
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Progress tracking
    progress_updates = []
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append({
            "status": status,
            "message": message
        })
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "processing": {"batch_size": 1, "max_retries": 3, "retry_failed": True},
            "performance": {"rate_limit_delay": 0.01},
            "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
            "preprocessing": {"lowercase": False, "remove_punctuation": False}
        }):
            # Test data
            test_terms = [{"id": "GO:0001", "name": "test term 1", "definition": "test def 1"}]
            
            await manager.create_and_load_ontology_collection(
                "test_collection",
                test_terms,
                "test_api_key",
                progress_callback
            )
            
            # Verify retries happened
            assert add_object_call_count == 3  # Failed twice, succeeded on third
            
            # Check for rate limit status in progress updates
            rate_limit_updates = [u for u in progress_updates if u["status"] == "rate_limited"]
            assert len(rate_limit_updates) >= 1


@pytest.mark.asyncio
async def test_batch_partial_failure_handling():
    """Test handling of partial batch failures."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Track which terms get added
    successful_terms = []
    
    def mock_add_object(obj):
        # Fail for terms with id ending in 2
        if obj["term_id"].endswith("2"):
            raise Exception("Failed to add term")
        successful_terms.append(obj["term_id"])
        return True
    
    mock_batch.add_object = mock_add_object
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    # Track stats
    final_stats = {}
    
    def progress_callback(status, percentage, message, extra_data):
        if extra_data:
            final_stats.update(extra_data)
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "processing": {"batch_size": 5, "max_retries": 0, "retry_failed": False},
            "performance": {"rate_limit_delay": 0},
            "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
            "preprocessing": {"lowercase": False, "remove_punctuation": False}
        }):
            # Test data with mix of success/failure
            test_terms = [
                {"id": "GO:0001", "name": "test term 1", "definition": "test def 1"},
                {"id": "GO:0002", "name": "test term 2", "definition": "test def 2"},  # Will fail
                {"id": "GO:0003", "name": "test term 3", "definition": "test def 3"},
                {"id": "GO:0012", "name": "test term 12", "definition": "test def 12"},  # Will fail
                {"id": "GO:0004", "name": "test term 4", "definition": "test def 4"}
            ]
            
            await manager.create_and_load_ontology_collection(
                "test_collection",
                test_terms,
                "test_api_key",
                progress_callback
            )
            
            # Verify correct terms succeeded
            assert "GO:0001" in successful_terms
            assert "GO:0002" not in successful_terms
            assert "GO:0003" in successful_terms
            assert "GO:0012" not in successful_terms
            assert "GO:0004" in successful_terms
            
            # Check final stats
            assert final_stats["processed_terms"] == 3
            assert final_stats["failed_terms"] == 2
            assert final_stats["total_terms"] == 5


@pytest.mark.asyncio
async def test_cancellation_during_embedding():
    """Test cancellation mechanism during embedding generation."""
    manager = OntologyManager()
    
    # Mock Weaviate client
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_batch = MagicMock()
    
    # Track progress
    progress_updates = []
    cancelled = False
    terms_processed = 0
    
    def mock_add_object(obj):
        nonlocal terms_processed
        terms_processed += 1
        # Simulate slow processing
        time.sleep(0.01)
        return True
    
    mock_batch.add_object = mock_add_object
    mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
    mock_client.collections.get.return_value = mock_collection
    mock_client.collections.create = MagicMock()
    mock_client.is_ready.return_value = True
    
    def progress_callback(status, percentage, message, extra_data):
        progress_updates.append({
            "status": status,
            "percentage": percentage,
            "message": message
        })
    
    def cancellation_check():
        nonlocal cancelled
        # Cancel after processing 3 terms
        if terms_processed >= 3:
            cancelled = True
        return cancelled
    
    with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "processing": {"batch_size": 2, "max_retries": 1, "retry_failed": True},
            "performance": {"rate_limit_delay": 0},
            "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
            "preprocessing": {"lowercase": False, "remove_punctuation": False}
        }):
            # Test data with many terms
            test_terms = [
                {"id": f"GO:{i:04d}", "name": f"test term {i}", "definition": f"test def {i}"}
                for i in range(1, 11)  # 10 terms
            ]
            
            await manager.create_and_load_ontology_collection(
                "test_collection",
                test_terms,
                "test_api_key",
                progress_callback,
                cancellation_check
            )
            
            # Verify cancellation happened
            assert cancelled
            assert terms_processed < 10  # Should not process all terms
            
            # Check for cancelled status in progress
            cancelled_updates = [u for u in progress_updates if u["status"] == "cancelled"]
            assert len(cancelled_updates) > 0
            assert "cancelled by user" in cancelled_updates[0]["message"].lower()


@pytest.mark.asyncio
async def test_background_task_error_handling():
    """Test error handling in the background embedding task."""
    from fastapi import HTTPException
    
    # Test file not found
    with patch('os.path.exists', return_value=False):
        embedding_progress_store.clear()
        
        with pytest.raises(HTTPException) as exc_info:
            await _generate_embeddings_only("DOID", "test_collection")
        
        assert exc_info.value.status_code == 404
        assert "DOID_embeddings" in embedding_progress_store
        assert embedding_progress_store["DOID_embeddings"]["status"] == "failed"


@pytest.mark.asyncio
async def test_embeddings_config_applied():
    """Test that embeddings configuration is properly applied."""
    manager = OntologyManager()
    
    # Mock the _build_searchable_text to verify it's called with correct config
    build_text_calls = []
    
    def mock_build_searchable_text(term_data):
        # Capture the current config state
        build_text_calls.append({
            "vectorize_fields": EMBEDDINGS_CONFIG.get("vectorize_fields", {}),
            "preprocessing": EMBEDDINGS_CONFIG.get("preprocessing", {})
        })
        return "test searchable text"
    
    with patch.object(manager, '_build_searchable_text', side_effect=mock_build_searchable_text):
        # Create mock client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        mock_batch.add_object = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.create = MagicMock()
        mock_client.is_ready.return_value = True
        
        with patch.object(manager, 'get_weaviate_client', return_value=mock_client):
            with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
                "vectorize_fields": {"name": False, "definition": True, "synonyms": False},
                "preprocessing": {"lowercase": True, "remove_punctuation": True},
                "processing": {"batch_size": 1}
            }):
                test_terms = [{"id": "GO:0001", "name": "test", "definition": "test def"}]
                
                await manager.create_and_load_ontology_collection(
                    "test_collection",
                    test_terms,
                    "test_api_key"
                )
                
                # Verify config was used
                assert len(build_text_calls) == 1
                assert build_text_calls[0]["vectorize_fields"] == {"name": False, "definition": True, "synonyms": False}
                assert build_text_calls[0]["preprocessing"] == {"lowercase": True, "remove_punctuation": True}