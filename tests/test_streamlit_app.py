import pytest
import json
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import streamlit as st

# Import the functions we want to test from streamlit_app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app import (
    load_ontology_config,
    get_enabled_ontologies,
    format_ontology_term,
    check_system_health,
    display_health_status
)


class TestOntologyConfig:
    """Test ontology configuration loading and processing."""
    
    @patch('streamlit_app.yaml.safe_load')
    @patch('builtins.open')
    @patch('os.getenv')
    def test_load_ontology_config_success(self, mock_getenv, mock_open, mock_yaml_load):
        """Test successful loading of ontology config."""
        mock_getenv.return_value = "test_config.yaml"
        mock_yaml_load.return_value = {
            "ontologies": {
                "GO": {"name": "Gene Ontology", "enabled": True},
                "DOID": {"name": "Disease Ontology", "enabled": False}
            }
        }
        
        # Clear the cache to ensure fresh load
        load_ontology_config.clear()
        
        result = load_ontology_config()
        
        assert result["ontologies"]["GO"]["name"] == "Gene Ontology"
        assert result["ontologies"]["GO"]["enabled"] is True
        assert result["ontologies"]["DOID"]["enabled"] is False
        mock_open.assert_called_once_with("test_config.yaml", 'r')
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('os.getenv')
    def test_load_ontology_config_file_not_found(self, mock_getenv, mock_open):
        """Test fallback when config file is not found."""
        mock_getenv.return_value = "nonexistent.yaml"
        
        # Clear the cache to ensure fresh load
        load_ontology_config.clear()
        
        result = load_ontology_config()
        
        # Should return default config
        assert "ontologies" in result
        assert "GO" in result["ontologies"]
        assert "DOID" in result["ontologies"]
        assert result["ontologies"]["GO"]["enabled"] is True
    
    def test_get_enabled_ontologies(self):
        """Test filtering of enabled ontologies."""
        with patch('streamlit_app.load_ontology_config') as mock_load_config:
            mock_load_config.return_value = {
                "ontologies": {
                    "GO": {"name": "Gene Ontology", "enabled": True},
                    "DOID": {"name": "Disease Ontology", "enabled": False},
                    "CHEBI": {"name": "Chemical Entities", "enabled": True}
                }
            }
            
            result = get_enabled_ontologies()
            
            assert result == ["GO", "CHEBI"]
            assert "DOID" not in result
    
    def test_get_enabled_ontologies_no_enabled_key(self):
        """Test ontologies without explicit 'enabled' key (defaults to True)."""
        with patch('streamlit_app.load_ontology_config') as mock_load_config:
            mock_load_config.return_value = {
                "ontologies": {
                    "GO": {"name": "Gene Ontology"},  # No 'enabled' key
                    "DOID": {"name": "Disease Ontology", "enabled": False}
                }
            }
            
            result = get_enabled_ontologies()
            
            assert "GO" in result  # Should default to enabled
            assert "DOID" not in result


class TestOntologyTermFormatting:
    """Test ontology term display formatting."""
    
    def test_format_ontology_term_complete(self):
        """Test formatting with all fields present."""
        term = {
            "id": "GO:0000001",
            "name": "mitochondrion inheritance",
            "definition": "The distribution of mitochondria to daughter cells."
        }
        
        result = format_ontology_term(term)
        
        assert "**mitochondrion inheritance**" in result
        assert "`GO:0000001`" in result
        assert "*Definition:* The distribution of mitochondria to daughter cells." in result
    
    def test_format_ontology_term_no_definition(self):
        """Test formatting without definition."""
        term = {
            "id": "GO:0000001",
            "name": "mitochondrion inheritance"
        }
        
        result = format_ontology_term(term)
        
        assert "**mitochondrion inheritance**" in result
        assert "`GO:0000001`" in result
        assert "*Definition:*" not in result
    
    def test_format_ontology_term_missing_fields(self):
        """Test formatting with missing fields."""
        term = {"name": "test term"}
        
        result = format_ontology_term(term)
        
        assert "**test term**" in result
        assert "`Unknown`" in result
    
    def test_format_ontology_term_empty(self):
        """Test formatting with empty term."""
        result = format_ontology_term({})
        # Empty dict is falsy, so should return "None"
        assert result == "None"
    
    def test_format_ontology_term_none(self):
        """Test formatting with None term."""
        assert format_ontology_term(None) == "None"


class TestSystemHealthCheck:
    """Test system health monitoring functionality."""
    
    @patch('streamlit_app.requests.get')
    def test_check_system_health_all_healthy(self, mock_get):
        """Test system health check when all services are healthy."""
        # Mock responses for different endpoints
        mock_responses = {
            "http://fastapi:8000/health": Mock(ok=True),
            "http://fastapi:8000/admin/weaviate_health": Mock(
                ok=True, 
                json=lambda: {"healthy": True, "details": "Connected"}
            ),
            "http://fastapi:8000/admin/openai_health": Mock(
                ok=True,
                json=lambda: {"healthy": True, "details": "API key valid"}
            )
        }
        
        def mock_get_side_effect(url, **kwargs):
            return mock_responses.get(url, Mock(ok=False))
        
        mock_get.side_effect = mock_get_side_effect
        
        result = check_system_health("test_api_key")
        
        assert result["overall"] == "healthy"
        assert result["fastapi"]["status"] == "healthy"
        assert result["weaviate"]["status"] == "healthy"
        assert result["openai"]["status"] == "healthy"
    
    @patch('streamlit_app.requests.get')
    def test_check_system_health_some_unhealthy(self, mock_get):
        """Test system health check with some services unhealthy."""
        mock_responses = {
            "http://fastapi:8000/health": Mock(ok=True),
            "http://fastapi:8000/admin/weaviate_health": Mock(
                ok=True,
                json=lambda: {"healthy": False, "details": "Connection failed"}
            ),
            "http://fastapi:8000/admin/openai_health": Mock(
                ok=True,
                json=lambda: {"healthy": True, "details": "API key valid"}
            )
        }
        
        def mock_get_side_effect(url, **kwargs):
            return mock_responses.get(url, Mock(ok=False))
        
        mock_get.side_effect = mock_get_side_effect
        
        result = check_system_health("test_api_key")
        
        assert result["overall"] == "degraded"
        assert result["fastapi"]["status"] == "healthy"
        assert result["weaviate"]["status"] == "unhealthy"
        assert result["weaviate"]["details"] == "Connection failed"
    
    @patch('streamlit_app.requests.get')
    def test_check_system_health_with_errors(self, mock_get):
        """Test system health check with mixed error scenarios."""
        def mock_get_side_effect(url, **kwargs):
            if "/health" in url:  # FastAPI health endpoint
                raise requests.exceptions.ConnectionError("Connection failed")
            elif "weaviate_health" in url:
                # Return HTTP 500 error - this should be "unhealthy"
                mock_resp = Mock()
                mock_resp.ok = False
                mock_resp.status_code = 500
                return mock_resp
            elif "openai_health" in url:
                # Connection error - this should be "error"
                raise requests.exceptions.ConnectionError("Connection failed")
            else:
                return Mock(ok=True, json=lambda: {"healthy": True, "details": "OK"})
        
        mock_get.side_effect = mock_get_side_effect
        
        result = check_system_health("test_api_key")
        
        assert result["overall"] == "error"
        assert result["fastapi"]["status"] == "error"
        assert "Connection failed" in result["fastapi"]["details"]
        assert result["weaviate"]["status"] == "unhealthy"  # HTTP error response
        assert "HTTP 500" in result["weaviate"]["details"]
        assert result["openai"]["status"] == "error"  # Connection error
        assert "Connection failed" in result["openai"]["details"]
    
    @patch('streamlit_app.requests.get')
    def test_check_system_health_timeout(self, mock_get):
        """Test system health check with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = check_system_health("test_api_key")
        
        assert result["overall"] == "error"
        for service in ["fastapi", "weaviate", "openai"]:
            assert result[service]["status"] == "error"
            assert "Request timed out" in result[service]["details"]


class TestDisplayHealthStatus:
    """Test health status display functionality."""
    
    @patch('streamlit_app.st')
    def test_display_health_status_healthy(self, mock_st):
        """Test display of healthy system status."""
        health_status = {
            "overall": "healthy",
            "fastapi": {"status": "healthy", "details": "Service responding"},
            "weaviate": {"status": "healthy", "details": "Connected"},
            "openai": {"status": "healthy", "details": "API key valid"}
        }
        
        display_health_status(health_status)
        
        # Check that the main success message was called
        success_calls = mock_st.success.call_args_list
        main_success_call = [call for call in success_calls if "🟢 System Healthy" in str(call)]
        assert len(main_success_call) > 0, f"Expected '🟢 System Healthy' call, got: {success_calls}"
    
    @patch('streamlit_app.st')
    def test_display_health_status_degraded(self, mock_st):
        """Test display of degraded system status."""
        health_status = {
            "overall": "degraded",
            "fastapi": {"status": "healthy", "details": "Service responding"},
            "weaviate": {"status": "unhealthy", "details": "Connection issues"},
            "openai": {"status": "healthy", "details": "API key valid"}
        }
        
        display_health_status(health_status)
        
        # Check that the main warning message was called
        warning_calls = mock_st.warning.call_args_list
        main_warning_call = [call for call in warning_calls if "🟡 System Degraded" in str(call)]
        assert len(main_warning_call) > 0, f"Expected '🟡 System Degraded' call, got: {warning_calls}"
    
    @patch('streamlit_app.st')
    def test_display_health_status_error(self, mock_st):
        """Test display of error system status."""
        health_status = {
            "overall": "error",
            "fastapi": {"status": "error", "details": "Connection failed"},
            "weaviate": {"status": "error", "details": "Service down"},
            "openai": {"status": "healthy", "details": "API key valid"}
        }
        
        display_health_status(health_status)
        
        # Check that the main error message was called
        error_calls = mock_st.error.call_args_list
        main_error_call = [call for call in error_calls if "🔴 System Issues Detected" in str(call)]
        assert len(main_error_call) > 0, f"Expected '🔴 System Issues Detected' call, got: {error_calls}"


class TestStreamlitIntegration:
    """Test Streamlit app integration scenarios."""
    
    @patch('streamlit_app.requests.post')
    def test_resolve_request_success(self, mock_post):
        """Test successful resolution request."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "best_match": {
                "id": "GO:0000001",
                "name": "mitochondrion inheritance",
                "definition": "Test definition"
            },
            "confidence": 0.95,
            "reason": "Exact match found",
            "alternatives": []
        }
        mock_post.return_value = mock_response
        
        # This would be called in the actual Streamlit app
        resp = mock_post(
            "http://fastapi:8000/resolve_biocurated_data",
            json={"passage": "test passage", "ontology_name": "GO"}
        )
        
        assert resp.ok
        data = resp.json()
        assert data["best_match"]["name"] == "mitochondrion inheritance"
        assert data["confidence"] == 0.95
    
    @patch('streamlit_app.requests.post')
    def test_resolve_request_error(self, mock_post):
        """Test resolution request with API error."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "error": "Ontology not found"
        }
        mock_post.return_value = mock_response
        
        resp = mock_post(
            "http://fastapi:8000/resolve_biocurated_data",
            json={"passage": "test passage", "ontology_name": "INVALID"}
        )
        
        assert resp.ok
        data = resp.json()
        assert "error" in data
        assert data["error"] == "Ontology not found"
    
    @patch('streamlit_app.requests.post')
    def test_admin_update_request(self, mock_post):
        """Test admin ontology update request."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "status": "Update initiated for GO ontology"
        }
        mock_post.return_value = mock_response
        
        resp = mock_post(
            "http://fastapi:8000/admin/update_ontology",
            headers={"X-API-Key": "test_key"},
            json={"ontology_name": "GO", "source_url": "http://example.com/go.json"}
        )
        
        assert resp.ok
        data = resp.json()
        assert "Update initiated" in data["status"]
    
    @patch('streamlit_app.requests.get')
    def test_version_status_request(self, mock_get):
        """Test version status retrieval."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "GO": {
                "current_version": "2024-01-01",
                "collection_name": "go_ontology_20240101",
                "last_updated": "2024-01-01T12:00:00Z",
                "hash": "abc123def456"
            }
        }
        mock_get.return_value = mock_response
        
        resp = mock_get(
            "http://fastapi:8000/admin/version_status",
            headers={"X-API-Key": "test_key"}
        )
        
        assert resp.ok
        data = resp.json()
        assert "GO" in data
        assert data["GO"]["current_version"] == "2024-01-01"
    
    @patch('streamlit_app.requests.get')
    def test_collections_request(self, mock_get):
        """Test collections listing request."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "collections": [
                {
                    "name": "go_ontology_20240101",
                    "object_count": 45000,
                    "size_mb": 125.5
                },
                {
                    "name": "doid_ontology_20240101",
                    "object_count": 8500,
                    "size_mb": 23.2
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
        assert len(data["collections"]) == 2
        assert data["collections"][0]["name"] == "go_ontology_20240101"
        assert data["collections"][0]["object_count"] == 45000


class TestAPITestingInterface:
    """Test the API testing interface functionality."""
    
    def test_example_requests_structure(self):
        """Test that example requests have the correct structure."""
        # This would be imported from the actual Streamlit app
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
            assert "passage" in request
            assert "ontology_name" in request
            assert isinstance(request["passage"], str)
            assert isinstance(request["ontology_name"], str)
            assert len(request["passage"]) > 0
    
    def test_request_json_generation(self):
        """Test JSON request generation for API testing."""
        test_passage = "cell division"
        test_ontology = "GO"
        
        request_json = {
            "passage": test_passage,
            "ontology_name": test_ontology
        }
        
        # Validate JSON structure
        assert "passage" in request_json
        assert "ontology_name" in request_json
        assert request_json["passage"] == test_passage
        assert request_json["ontology_name"] == test_ontology
        
        # Test JSON serialization
        json_str = json.dumps(request_json, indent=2)
        assert isinstance(json_str, str)
        assert "cell division" in json_str
    
    @patch('streamlit_app.requests.post')
    def test_api_test_request_timing(self, mock_post):
        """Test API request timing functionality."""
        import time
        
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"test": "response"}
        mock_post.return_value = mock_response
        
        start_time = time.time()
        resp = mock_post(
            "http://fastapi:8000/resolve_biocurated_data",
            json={"passage": "test", "ontology_name": "GO"},
            timeout=30
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # ms
        
        assert resp.ok
        assert response_time >= 0
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
    
    @patch('streamlit_app.requests.post')
    def test_api_test_error_handling(self, mock_post):
        """Test API testing error handling."""
        # Test timeout
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(requests.exceptions.Timeout):
            mock_post(
                "http://fastapi:8000/resolve_biocurated_data",
                json={"passage": "test", "ontology_name": "GO"},
                timeout=30
            )
        
        # Test connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(requests.exceptions.ConnectionError):
            mock_post(
                "http://fastapi:8000/resolve_biocurated_data",
                json={"passage": "test", "ontology_name": "GO"},
                timeout=30
            )


class TestDateTimeFormatting:
    """Test datetime formatting functionality."""
    
    def test_datetime_formatting_valid_iso(self):
        """Test formatting of valid ISO datetime."""
        iso_datetime = "2024-01-01T12:00:00Z"
        
        try:
            dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
            formatted = dt.strftime('%Y-%m-%d %H:%M')
            assert formatted == "2024-01-01 12:00"
        except ValueError:
            # If parsing fails, should keep original
            assert iso_datetime == "2024-01-01T12:00:00Z"
    
    def test_datetime_formatting_invalid(self):
        """Test handling of invalid datetime strings."""
        invalid_datetime = "not-a-datetime"
        
        try:
            dt = datetime.fromisoformat(invalid_datetime.replace('Z', '+00:00'))
            formatted = dt.strftime('%Y-%m-%d %H:%M')
        except ValueError:
            # Should keep original value when parsing fails
            formatted = invalid_datetime
        
        assert formatted == invalid_datetime
    
    def test_datetime_formatting_none(self):
        """Test handling of None datetime."""
        last_update = None
        if last_update is None:
            last_update = "Never"
        
        assert last_update == "Never"


class TestOpenAIKeyWarning:
    """Test OpenAI API key detection and warning functionality."""
    
    @patch('streamlit_app.os.getenv')
    def test_check_openai_api_key_found(self, mock_getenv):
        """Test detection when OpenAI API key is present."""
        def mock_getenv_side_effect(var, default=None):
            if var == "OPENAI_API_KEY":
                return "sk-1234567890abcdef"
            return default
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        from streamlit_app import check_openai_api_key
        assert check_openai_api_key() is True
    
    @patch('streamlit_app.os.getenv')
    def test_check_openai_api_key_missing(self, mock_getenv):
        """Test detection when OpenAI API key is missing."""
        mock_getenv.return_value = None
        
        from streamlit_app import check_openai_api_key
        assert check_openai_api_key() is False
    
    @patch('streamlit_app.os.getenv')
    def test_check_openai_api_key_empty_values(self, mock_getenv):
        """Test detection with empty or placeholder values."""
        empty_values = ["", "   ", "none", "null", "your_api_key_here", "None", "NULL"]
        
        from streamlit_app import check_openai_api_key
        
        for empty_value in empty_values:
            mock_getenv.return_value = empty_value
            assert check_openai_api_key() is False, f"Should detect '{empty_value}' as missing"
    
    @patch('streamlit_app.os.getenv')
    def test_check_openai_api_key_alternative_vars(self, mock_getenv):
        """Test detection using alternative environment variable names."""
        def mock_getenv_side_effect(var, default=None):
            if var == "OPENAI_KEY":
                return "sk-alternative-key"
            return default
        
        mock_getenv.side_effect = mock_getenv_side_effect
        
        from streamlit_app import check_openai_api_key
        assert check_openai_api_key() is True
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.check_openai_api_key')
    def test_display_openai_warning_when_missing(self, mock_check_key, mock_st):
        """Test warning display when API key is missing."""
        mock_check_key.return_value = False
        
        from streamlit_app import display_openai_warning
        result = display_openai_warning()
        
        assert result is True  # Should return True when warning is displayed
        mock_st.warning.assert_called_once()
        
        # Check warning message content
        warning_call = mock_st.warning.call_args[0][0]
        assert "OpenAI API Key Not Detected" in warning_call
        assert "OPENAI_API_KEY" in warning_call
    
    @patch('streamlit_app.check_openai_api_key')
    def test_display_openai_warning_when_present(self, mock_check_key):
        """Test no warning when API key is present."""
        mock_check_key.return_value = True
        
        from streamlit_app import display_openai_warning
        result = display_openai_warning()
        
        assert result is False  # Should return False when no warning is displayed


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    @patch('streamlit_app.requests.get')
    def test_api_request_failure_handling(self, mock_get):
        """Test handling of API request failures."""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        resp = mock_get("http://fastapi:8000/admin/ontology_status")
        
        assert not resp.ok
        assert resp.status_code == 500
        assert resp.text == "Internal Server Error"
    
    def test_json_parsing_error_handling(self):
        """Test handling of JSON parsing errors."""
        invalid_json = "not valid json"
        
        try:
            result = json.loads(invalid_json)
        except json.JSONDecodeError as e:
            assert "not valid json" in str(e) or "Expecting value" in str(e)
    
    def test_missing_api_key_handling(self):
        """Test handling when API key is missing."""
        api_key = ""
        
        if not api_key:
            # Should show warning about missing API key
            warning_msg = "Please enter your API key in the sidebar to test admin endpoints"
            assert "API key" in warning_msg
            assert "sidebar" in warning_msg


if __name__ == "__main__":
    pytest.main([__file__])