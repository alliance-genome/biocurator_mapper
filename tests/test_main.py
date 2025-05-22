import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app, get_ontology_config, get_nested_value, _perform_ontology_update


class MockHttpResponse:
    """Mock aiohttp response context manager for testing."""
    
    def __init__(self, response_mock):
        self._response = response_mock
    
    async def __aenter__(self):
        return self._response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestHelperFunctions:
    """Test utility functions in main.py."""

    def test_get_ontology_config_existing(self):
        """Test getting config for existing ontology."""
        mock_config = {
            "ontologies": {
                "GO": {"name": "Gene Ontology", "enabled": True},
                "DOID": {"name": "Disease Ontology", "enabled": True}
            }
        }
        
        with patch("app.main.ONTOLOGY_CONFIG", mock_config):
            config = get_ontology_config("GO")
            assert config["name"] == "Gene Ontology"
            assert config["enabled"] is True

    def test_get_ontology_config_nonexistent(self):
        """Test getting config for non-existent ontology."""
        mock_config = {"ontologies": {}}
        
        with patch("app.main.ONTOLOGY_CONFIG", mock_config):
            config = get_ontology_config("NONEXISTENT")
            assert config == {}

    def test_get_nested_value_success(self):
        """Test getting nested value when path exists."""
        data = {
            "meta": {
                "definition": {
                    "val": "test definition"
                }
            }
        }
        
        result = get_nested_value(data, ["meta", "definition", "val"])
        assert result == "test definition"

    def test_get_nested_value_missing_path(self):
        """Test getting nested value when path doesn't exist."""
        data = {"meta": {}}
        
        result = get_nested_value(data, ["meta", "definition", "val"], "default")
        assert result == "default"

    def test_get_nested_value_empty_data(self):
        """Test getting nested value from empty data."""
        result = get_nested_value({}, ["meta", "definition", "val"])
        assert result == ""


class TestAPI:
    """Test API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_get_ontology_config_endpoint(self):
        """Test the ontology config endpoint."""
        mock_config = {
            "ontologies": {
                "GO": {
                    "name": "Gene Ontology",
                    "description": "GO description",
                    "enabled": True,
                    "default_source_url": "http://example.com/go.json"
                }
            }
        }
        
        with patch("app.main.ONTOLOGY_CONFIG", mock_config):
            response = self.client.get("/ontology_config")
            
        assert response.status_code == 200
        data = response.json()
        assert "ontologies" in data
        assert "GO" in data["ontologies"]
        assert data["ontologies"]["GO"]["name"] == "Gene Ontology"

    @patch("app.main.ontology_manager")
    @patch("app.main.searcher")
    @patch("app.main.matcher")
    def test_resolve_biocurated_data_success(self, mock_llm, mock_searcher, mock_ontology_manager):
        """Test successful biocurated data resolution."""
        # Mock searcher results
        mock_searcher.search_ontology = AsyncMock(return_value=[
            {"id": "GO:0001", "name": "Test Term", "definition": "Test definition"}
        ])
        
        # Mock LLM matcher results
        mock_llm.select_best_match = AsyncMock(return_value={
            "id": "GO:0001", 
            "name": "Test Term",
            "confidence": 0.9,
            "reason": "High semantic similarity"
        })
        
        # Mock ontology manager
        mock_ontology_manager.get_current_ontology_version.return_value = "GO_collection_test"
        
        response = self.client.post(
            "/resolve_biocurated_data",
            json={"passage": "test passage", "ontology_name": "GO"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "best_match" in data
        assert data["best_match"]["id"] == "GO:0001"

    def test_resolve_biocurated_data_missing_fields(self):
        """Test biocurated data resolution with missing fields."""
        response = self.client.post(
            "/resolve_biocurated_data",
            json={"passage": "test passage"}  # Missing ontology_name
        )
        
        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
class TestOntologyUpdate:
    """Test ontology update functionality."""

    @patch("app.main.version_manager")
    @patch("app.main.ontology_manager")
    @patch("app.main.config_updater")
    @patch("aiohttp.ClientSession")
    async def test_perform_ontology_update_success(self, mock_session, mock_config_updater, mock_ontology_manager, mock_version_manager):
        """Test successful ontology update."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "graphs": [{
                "nodes": [
                    {
                        "id": "http://example.com/GO_0001",
                        "lbl": "Test Term",
                        "meta": {"definition": {"val": "Test definition"}}
                    }
                ]
            }]
        }
        
        # Setup proper aiohttp mocking
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = MockHttpResponse(mock_response)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock version manager and ontology manager
        mock_version_manager.compare_versions.return_value = (True, None, {"version_date": "2025-03-16"})
        mock_ontology_manager.create_and_load_ontology_collection = AsyncMock(return_value=None)
        
        with patch("app.main.ONTOLOGY_CONFIG", {
            "ontologies": {"GO": {"id_format": {"prefix_replacement": {"_": ":"}}}},
            "settings": {"json_parsing": {
                "graphs_key": "graphs",
                "nodes_key": "nodes", 
                "id_key": "id",
                "label_key": "lbl",
                "definition_path": ["meta", "definition", "val"]
            }}
        }):
            await _perform_ontology_update("GO", "http://example.com/go.json")
        
        # Verify calls
        mock_ontology_manager.create_and_load_ontology_collection.assert_called_once()
        mock_config_updater.update_ontology_version.assert_called_once()

    @patch("app.main.version_manager")
    @patch("aiohttp.ClientSession")
    async def test_perform_ontology_update_http_error(self, mock_session, mock_version_manager):
        """Test ontology update with HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Setup proper aiohttp mocking  
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = MockHttpResponse(mock_response)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="Failed to download ontology"):
            await _perform_ontology_update("GO", "http://example.com/invalid.json")