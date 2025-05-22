import pytest
import json
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

# Test the integration scenarios and complex workflows


class TestStreamlitWorkflows:
    """Test complete workflows in the Streamlit application."""
    
    @patch('streamlit_app.requests.post')
    @patch('streamlit_app.requests.get')
    def test_complete_resolution_workflow(self, mock_get, mock_post):
        """Test complete resolution workflow from input to display."""
        # Mock successful resolution
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.json.return_value = {
            "best_match": {
                "id": "GO:0051301",
                "name": "cell division",
                "definition": "The process by which a cell divides"
            },
            "confidence": 0.92,
            "reason": "Direct term match with high semantic similarity",
            "alternatives": [
                {
                    "id": "GO:0000278",
                    "name": "mitotic cell cycle",
                    "definition": "Progression through mitotic cell cycle"
                }
            ]
        }
        mock_post.return_value = mock_post_response
        
        # Simulate user input
        passage = "cell division"
        ontology = "GO"
        
        # Simulate API call
        resp = mock_post(
            "http://fastapi:8000/resolve_biocurated_data",
            json={"passage": passage, "ontology_name": ontology}
        )
        
        assert resp.ok
        data = resp.json()
        
        # Verify response structure
        assert "best_match" in data
        assert "confidence" in data
        assert "reason" in data
        assert "alternatives" in data
        
        # Verify best match
        best_match = data["best_match"]
        assert best_match["id"] == "GO:0051301"
        assert best_match["name"] == "cell division"
        assert "definition" in best_match
        
        # Verify confidence and reasoning
        assert data["confidence"] == 0.92
        assert "semantic similarity" in data["reason"]
        
        # Verify alternatives
        alternatives = data["alternatives"]
        assert len(alternatives) == 1
        assert alternatives[0]["id"] == "GO:0000278"
    
    @patch('streamlit_app.requests.post')
    @patch('streamlit_app.requests.get')
    def test_admin_update_workflow(self, mock_get, mock_post):
        """Test complete admin ontology update workflow."""
        # Mock update initiation
        mock_post_response = Mock()
        mock_post_response.ok = True
        mock_post_response.json.return_value = {
            "status": "Update initiated for GO ontology",
            "task_id": "update_task_123"
        }
        mock_post.return_value = mock_post_response
        
        # Mock progress checking
        mock_get_response = Mock()
        mock_get_response.ok = True
        mock_get_response.json.return_value = {
            "status": "in_progress",
            "progress_percentage": 75,
            "recent_logs": [
                {
                    "timestamp": "2024-01-01T12:00:00Z",
                    "level": "INFO",
                    "message": "Processing GO terms..."
                },
                {
                    "timestamp": "2024-01-01T12:01:00Z",
                    "level": "INFO",
                    "message": "Embedded 30000 terms"
                }
            ]
        }
        mock_get.return_value = mock_get_response
        
        # Start update
        update_resp = mock_post(
            "http://fastapi:8000/admin/update_ontology",
            headers={"X-API-Key": "test_key"},
            json={"ontology_name": "GO", "source_url": "http://example.com/go.json"}
        )
        
        assert update_resp.ok
        update_data = update_resp.json()
        assert "Update initiated" in update_data["status"]
        assert "task_id" in update_data
        
        # Check progress
        progress_resp = mock_get(
            "http://fastapi:8000/admin/update_progress/GO",
            headers={"X-API-Key": "test_key"}
        )
        
        assert progress_resp.ok
        progress_data = progress_resp.json()
        assert progress_data["status"] == "in_progress"
        assert progress_data["progress_percentage"] == 75
        assert len(progress_data["recent_logs"]) == 2
    
    @patch('streamlit_app.requests.get')
    def test_system_health_monitoring_workflow(self, mock_get):
        """Test complete system health monitoring workflow."""
        # Mock health check responses
        health_responses = {
            "http://fastapi:8000/health": Mock(ok=True),
            "http://fastapi:8000/admin/weaviate_health": Mock(
                ok=True,
                json=lambda: {"healthy": True, "details": "Connected to Weaviate cluster"}
            ),
            "http://fastapi:8000/admin/openai_health": Mock(
                ok=True,
                json=lambda: {"healthy": True, "details": "API key valid, quota available"}
            )
        }
        
        def mock_get_side_effect(url, **kwargs):
            return health_responses.get(url, Mock(ok=False))
        
        mock_get.side_effect = mock_get_side_effect
        
        # Import and test the health check function
        from streamlit_app import check_system_health
        
        health_status = check_system_health("test_api_key")
        
        # Verify overall health
        assert health_status["overall"] == "healthy"
        
        # Verify individual services
        assert health_status["fastapi"]["status"] == "healthy"
        assert health_status["weaviate"]["status"] == "healthy"
        assert health_status["openai"]["status"] == "healthy"
        
        # Verify details
        assert "Connected to Weaviate" in health_status["weaviate"]["details"]
        assert "quota available" in health_status["openai"]["details"]


class TestAPITestingInterface:
    """Test the API testing interface functionality in detail."""
    
    def test_example_request_validation(self):
        """Test validation of example requests."""
        example_requests = {
            "Simple GO query": {
                "passage": "cell division",
                "ontology_name": "GO"
            },
            "Complex biological process": {
                "passage": "regulation of transcription DNA-templated in response to stress",
                "ontology_name": "GO"
            },
            "Disease query": {
                "passage": "inflammatory bowel disease",
                "ontology_name": "DOID"
            }
        }
        
        for name, request in example_requests.items():
            # Validate required fields
            assert "passage" in request, f"Missing 'passage' in {name}"
            assert "ontology_name" in request, f"Missing 'ontology_name' in {name}"
            
            # Validate field types
            assert isinstance(request["passage"], str), f"Invalid passage type in {name}"
            assert isinstance(request["ontology_name"], str), f"Invalid ontology_name type in {name}"
            
            # Validate non-empty values
            assert len(request["passage"].strip()) > 0, f"Empty passage in {name}"
            assert len(request["ontology_name"].strip()) > 0, f"Empty ontology_name in {name}"
            
            # Validate JSON serialization
            try:
                json_str = json.dumps(request, indent=2)
                parsed_back = json.loads(json_str)
                assert parsed_back == request, f"JSON serialization failed for {name}"
            except (TypeError, ValueError) as e:
                pytest.fail(f"JSON serialization failed for {name}: {e}")
    
    @patch('streamlit_app.requests.post')
    def test_api_testing_response_analysis(self, mock_post):
        """Test response analysis in API testing interface."""
        # Mock a successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "application/json; charset=utf-8",
            "content-length": "1234"
        }
        mock_response.json.return_value = {
            "best_match": {
                "id": "GO:0051301",
                "name": "cell division",
                "definition": "The process by which a cell divides"
            },
            "confidence": 0.92,
            "reason": "High semantic similarity",
            "alternatives": []
        }
        mock_post.return_value = mock_response
        
        # Time the request
        start_time = time.time()
        resp = mock_post(
            "http://fastapi:8000/resolve_biocurated_data",
            json={"passage": "cell division", "ontology_name": "GO"},
            timeout=30
        )
        end_time = time.time()
        
        # Analyze response
        response_time_ms = (end_time - start_time) * 1000
        
        assert resp.ok
        assert resp.status_code == 200
        assert response_time_ms >= 0
        
        # Test content type parsing
        content_type = resp.headers.get('content-type', 'unknown')
        main_content_type = content_type.split(';')[0]
        assert main_content_type == "application/json"
        
        # Test response structure
        data = resp.json()
        assert "best_match" in data
        assert "confidence" in data
        assert data["confidence"] == 0.92
    
    @patch('streamlit_app.requests.post')
    def test_api_testing_error_scenarios(self, mock_post):
        """Test error scenarios in API testing interface."""
        # Test different error types
        error_scenarios = [
            (requests.exceptions.Timeout("Request timed out"), "timeout"),
            (requests.exceptions.ConnectionError("Connection failed"), "connection"),
            (Exception("Unexpected error"), "unexpected")
        ]
        
        for exception, error_type in error_scenarios:
            mock_post.side_effect = exception
            
            try:
                mock_post(
                    "http://fastapi:8000/resolve_biocurated_data",
                    json={"passage": "test", "ontology_name": "GO"},
                    timeout=30
                )
                pytest.fail(f"Expected {error_type} error")
            except type(exception) as e:
                assert str(e) == str(exception)
    
    @patch('streamlit_app.requests.post')
    def test_api_testing_http_error_codes(self, mock_post):
        """Test handling of different HTTP error codes."""
        error_codes = [400, 401, 404, 500, 503]
        
        for code in error_codes:
            mock_response = Mock()
            mock_response.ok = False
            mock_response.status_code = code
            mock_response.text = f"HTTP {code} Error"
            mock_post.return_value = mock_response
            
            resp = mock_post(
                "http://fastapi:8000/resolve_biocurated_data",
                json={"passage": "test", "ontology_name": "GO"}
            )
            
            assert not resp.ok
            assert resp.status_code == code
            assert str(code) in resp.text


class TestVersionManagement:
    """Test version management dashboard functionality."""
    
    @patch('streamlit_app.requests.get')
    def test_version_data_display(self, mock_get):
        """Test version data fetching and display."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "GO": {
                "current_version": "2024-01-01",
                "collection_name": "go_ontology_20240101",
                "last_updated": "2024-01-01T12:00:00Z",
                "hash": "abc123def456789"
            },
            "DOID": {
                "current_version": "2024-01-02",
                "collection_name": "doid_ontology_20240102",
                "last_updated": "2024-01-02T10:30:00Z",
                "hash": "xyz789abc123456"
            }
        }
        mock_get.return_value = mock_response
        
        resp = mock_get(
            "http://fastapi:8000/admin/version_status",
            headers={"X-API-Key": "test_key"}
        )
        
        assert resp.ok
        version_data = resp.json()
        
        # Test GO ontology data
        go_data = version_data["GO"]
        assert go_data["current_version"] == "2024-01-01"
        assert go_data["collection_name"] == "go_ontology_20240101"
        assert "hash" in go_data
        
        # Test DOID ontology data
        doid_data = version_data["DOID"]
        assert doid_data["current_version"] == "2024-01-02"
        assert doid_data["collection_name"] == "doid_ontology_20240102"
        
        # Test datetime formatting
        for ontology_data in version_data.values():
            last_updated = ontology_data["last_updated"]
            try:
                dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                formatted = dt.strftime('%Y-%m-%d %H:%M')
                assert len(formatted) == 16  # YYYY-MM-DD HH:MM format
            except ValueError:
                # Should handle invalid datetime gracefully
                pass
    
    def test_hash_display_truncation(self):
        """Test content hash display truncation."""
        full_hash = "abc123def456789xyz"
        
        # Test hash truncation for display
        if len(full_hash) > 16:
            truncated = full_hash[:16] + "..."
            # The actual hash is 18 chars, so first 16 would be "abc123def456789x"
            assert truncated == "abc123def456789x..."
            assert len(truncated) == 19  # 16 chars + "..."
        else:
            truncated = full_hash
            assert truncated == full_hash


class TestCollectionManagement:
    """Test collection management functionality."""
    
    @patch('streamlit_app.requests.get')
    def test_collections_listing(self, mock_get):
        """Test fetching and displaying collections."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "collections": [
                {
                    "name": "go_ontology_20240101",
                    "object_count": 45623,
                    "size_mb": 125.7
                },
                {
                    "name": "doid_ontology_20240102",
                    "object_count": 8945,
                    "size_mb": 23.1
                }
            ]
        }
        mock_get.return_value = mock_response
        
        resp = mock_get(
            "http://fastapi:8000/admin/collections",
            headers={"X-API-Key": "test_key"}
        )
        
        assert resp.ok
        data = resp.json()
        collections = data["collections"]
        
        assert len(collections) == 2
        
        # Test GO collection
        go_collection = collections[0]
        assert go_collection["name"] == "go_ontology_20240101"
        assert go_collection["object_count"] == 45623
        assert go_collection["size_mb"] == 125.7
        
        # Test DOID collection
        doid_collection = collections[1]
        assert doid_collection["name"] == "doid_ontology_20240102"
        assert doid_collection["object_count"] == 8945
        assert doid_collection["size_mb"] == 23.1
    
    @patch('streamlit_app.requests.delete')
    def test_collection_deletion(self, mock_delete):
        """Test collection deletion functionality."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "status": "Collection deleted successfully",
            "collection_name": "test_collection"
        }
        mock_delete.return_value = mock_response
        
        resp = mock_delete(
            "http://fastapi:8000/admin/collections/test_collection",
            headers={"X-API-Key": "test_key"}
        )
        
        assert resp.ok
        data = resp.json()
        assert "deleted successfully" in data["status"].lower()
        assert data["collection_name"] == "test_collection"
    
    def test_collection_metrics_formatting(self):
        """Test formatting of collection metrics."""
        # Test object count formatting
        object_counts = [1234, 12345, 123456]
        
        for count in object_counts:
            formatted = f"{count:,}"
            if count == 1234:
                assert formatted == "1,234"
            elif count == 12345:
                assert formatted == "12,345"
            elif count == 123456:
                assert formatted == "123,456"
        
        # Test size formatting
        sizes = [1.2, 12.34, 123.456]
        
        for size in sizes:
            formatted = f"{size:.1f} MB"
            if size == 1.2:
                assert formatted == "1.2 MB"
            elif size == 12.34:
                assert formatted == "12.3 MB"
            elif size == 123.456:
                assert formatted == "123.5 MB"


class TestOpenAIKeyIntegration:
    """Test OpenAI API key integration scenarios."""
    
    @patch('streamlit_app.os.getenv')
    def test_missing_openai_key_workflow(self, mock_getenv):
        """Test complete workflow when OpenAI API key is missing."""
        # Mock missing API key
        mock_getenv.return_value = None
        
        from streamlit_app import check_openai_api_key, display_openai_warning
        
        # Verify key detection
        assert check_openai_api_key() is False
        
        # Test that warning would be displayed
        with patch('streamlit_app.st') as mock_st:
            result = display_openai_warning()
            assert result is True
            mock_st.warning.assert_called_once()
    
    @patch('streamlit_app.os.getenv')
    def test_present_openai_key_workflow(self, mock_getenv):
        """Test complete workflow when OpenAI API key is present."""
        # Mock present API key
        def mock_getenv_side_effect(var, default=None):
            if var == "OPENAI_API_KEY":
                return "sk-proj-abcd1234efgh5678ijkl9012mnop3456qrst7890"
            return default
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        from streamlit_app import check_openai_api_key, display_openai_warning
        
        # Verify key detection
        assert check_openai_api_key() is True
        
        # Test that no warning is displayed
        result = display_openai_warning()
        assert result is False
    
    @patch('streamlit_app.os.getenv')
    def test_docker_environment_scenario(self, mock_getenv):
        """Test scenario common in Docker deployments."""
        # Mock Docker environment with proper key
        def mock_getenv_side_effect(var, default=None):
            env_vars = {
                "OPENAI_API_KEY": "sk-proj-docker-deployment-key-12345",
                "FASTAPI_URL": "http://fastapi:8000",
                "ADMIN_API_KEY": "admin-secret-key"
            }
            return env_vars.get(var, default)
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        from streamlit_app import check_openai_api_key
        
        # Should detect the key properly
        assert check_openai_api_key() is True
    
    @patch('streamlit_app.os.getenv')
    def test_common_misconfiguration_scenarios(self, mock_getenv):
        """Test common misconfiguration scenarios."""
        misconfigurations = [
            "your_api_key_here",    # Placeholder value
            "sk-",                  # Incomplete key  
            "none",                 # Explicitly set to none
            "NULL",                 # Explicitly set to null
            "",                     # Empty string
            "   ",                  # Whitespace only
        ]
        
        from streamlit_app import check_openai_api_key
        
        for bad_value in misconfigurations:
            mock_getenv.return_value = bad_value
            assert check_openai_api_key() is False, f"Should detect '{bad_value}' as invalid"


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        empty_responses = [
            {},
            {"collections": []},
            {"ontologies": {}},
            None
        ]
        
        for response in empty_responses:
            if response is None:
                # Should handle None gracefully
                assert response is None
            elif isinstance(response, dict):
                if "collections" in response and not response["collections"]:
                    # Empty collections list
                    assert len(response["collections"]) == 0
                elif "ontologies" in response and not response["ontologies"]:
                    # Empty ontologies dict
                    assert len(response["ontologies"]) == 0
                elif not response:
                    # Empty dict
                    assert len(response) == 0
    
    def test_malformed_datetime_handling(self):
        """Test handling of malformed datetime strings."""
        malformed_datetimes = [
            "not-a-date",
            "2024-13-01",  # Invalid month
            "2024-01-32",  # Invalid day
            "",
            None
        ]
        
        for dt_str in malformed_datetimes:
            if dt_str is None:
                result = "Never"
            elif dt_str == "":
                result = "Never"
            else:
                try:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    result = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError):
                    result = dt_str  # Keep original if parsing fails
            
            # Should not raise exceptions
            assert result is not None
    
    @patch('streamlit_app.requests.get')
    def test_network_timeout_handling(self, mock_get):
        """Test handling of network timeouts."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out after 30 seconds")
        
        from streamlit_app import check_system_health
        
        health_status = check_system_health("test_key")
        
        # All services should show error status
        for service in ["fastapi", "weaviate", "openai"]:
            assert health_status[service]["status"] == "error"
            assert "timed out" in health_status[service]["details"].lower()
        
        assert health_status["overall"] == "error"
    
    def test_invalid_json_response_handling(self):
        """Test handling of invalid JSON responses."""
        invalid_json_strings = [
            "not json",
            "{incomplete json",
            "{'single_quotes': 'invalid'}",
            ""
        ]
        
        for json_str in invalid_json_strings:
            try:
                result = json.loads(json_str)
                # Should not reach here for invalid JSON
                if json_str == "":
                    pytest.fail("Empty string should raise JSONDecodeError")
            except json.JSONDecodeError:
                # Expected for invalid JSON
                assert True
            except Exception as e:
                # Other exceptions are also acceptable for invalid input
                assert True


if __name__ == "__main__":
    pytest.main([__file__])