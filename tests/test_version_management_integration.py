import pytest
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

from app.main import _perform_ontology_update
from app.ontology_version_manager import OntologyVersionManager


class MockHttpResponse:
    """Mock aiohttp response context manager for testing."""
    
    def __init__(self, response_mock):
        self._response = response_mock
    
    async def __aenter__(self):
        return self._response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.mark.asyncio
class TestVersionManagementIntegration:
    """Test version management integration in main ontology update flow."""

    @patch("app.main.version_manager")
    @patch("app.main.ontology_manager")
    @patch("app.main.config_updater")
    @patch("aiohttp.ClientSession")
    async def test_ontology_update_skips_when_up_to_date(self, mock_session, mock_config_updater, mock_ontology_manager, mock_version_manager):
        """Test that ontology update is skipped when version is up to date."""
        # Mock HTTP response with GO data
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "meta": {"version": "2025-03-16"},
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
        
        # Mock session setup
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = MockHttpResponse(mock_response)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock version manager - indicating no update needed
        stored_info = {
            "collection_name": "GO_existing_collection",
            "version_info": {"version_date": "2025-03-16", "content_hash": "abc123"}
        }
        mock_version_manager.compare_versions.return_value = (False, stored_info, {"version_date": "2025-03-16"})
        
        with patch("app.main.ONTOLOGY_CONFIG", {
            "ontologies": {"GO": {"id_format": {"prefix_replacement": {"_": ":"}}}},
            "settings": {"json_parsing": {}}
        }):
            result = await _perform_ontology_update("GO", "http://example.com/go.json")
        
        # Verify that we use existing collection and skip embedding
        assert result == "GO_existing_collection"
        mock_ontology_manager.create_and_load_ontology_collection.assert_not_called()
        mock_config_updater.update_ontology_version.assert_called_once_with(
            "GO", "GO_existing_collection", "http://example.com/go.json"
        )

    @patch("app.main.version_manager")
    @patch("app.main.ontology_manager")
    @patch("app.main.config_updater")
    @patch("aiohttp.ClientSession")
    async def test_ontology_update_proceeds_when_version_changed(self, mock_session, mock_config_updater, mock_ontology_manager, mock_version_manager):
        """Test that ontology update proceeds when version has changed."""
        # Mock HTTP response with new GO data
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "meta": {"version": "2025-03-17"},
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
        
        # Mock session setup
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = MockHttpResponse(mock_response)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock version manager - indicating update needed
        stored_info = {
            "collection_name": "GO_old_collection",
            "version_info": {"version_date": "2025-03-16", "content_hash": "abc123"}
        }
        new_version_info = {"version_date": "2025-03-17", "content_hash": "def456"}
        mock_version_manager.compare_versions.return_value = (True, stored_info, new_version_info)
        
        # Mock ontology manager
        mock_ontology_manager.create_and_load_ontology_collection = AsyncMock(return_value=None)
        
        with patch("app.main.ONTOLOGY_CONFIG", {
            "ontologies": {"GO": {"id_format": {"prefix_replacement": {"_": ":"}}}},
            "settings": {"json_parsing": {}}
        }):
            result = await _perform_ontology_update("GO", "http://example.com/go.json")
        
        # Verify that new collection is created
        mock_ontology_manager.create_and_load_ontology_collection.assert_called_once()
        mock_version_manager.store_version_info.assert_called_once_with(
            "GO", new_version_info, result, "http://example.com/go.json"
        )
        mock_config_updater.update_ontology_version.assert_called_once()

    @patch("app.main.version_manager")
    @patch("app.main.ontology_manager")
    @patch("app.main.config_updater")
    @patch("aiohttp.ClientSession")
    async def test_ontology_update_first_time_setup(self, mock_session, mock_config_updater, mock_ontology_manager, mock_version_manager):
        """Test ontology update when no previous version exists."""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "meta": {"version": "2025-03-16"},
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
        
        # Mock session setup
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = MockHttpResponse(mock_response)
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Mock version manager - no stored info, first time setup
        new_version_info = {"version_date": "2025-03-16", "content_hash": "abc123"}
        mock_version_manager.compare_versions.return_value = (True, None, new_version_info)
        
        # Mock ontology manager
        mock_ontology_manager.create_and_load_ontology_collection = AsyncMock(return_value=None)
        
        with patch("app.main.ONTOLOGY_CONFIG", {
            "ontologies": {"GO": {"id_format": {"prefix_replacement": {"_": ":"}}}},
            "settings": {"json_parsing": {}}
        }):
            result = await _perform_ontology_update("GO", "http://example.com/go.json")
        
        # Verify that new collection is created for first time
        mock_ontology_manager.create_and_load_ontology_collection.assert_called_once()
        mock_version_manager.store_version_info.assert_called_once_with(
            "GO", new_version_info, result, "http://example.com/go.json"
        )

    @patch("app.main.version_manager")
    @patch("aiohttp.ClientSession")
    async def test_version_comparison_handles_download_errors(self, mock_session, mock_version_manager):
        """Test that version comparison gracefully handles download errors."""
        # Mock HTTP error
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = aiohttp.ClientError("Connection failed")
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        with pytest.raises(aiohttp.ClientError):
            await _perform_ontology_update("GO", "http://example.com/invalid.json")
        
        # Version manager should not be called if download fails
        mock_version_manager.compare_versions.assert_not_called()


class TestVersionManagerIntegration:
    """Test direct version manager functionality."""

    def test_version_manager_extract_go_version_info(self):
        """Test extracting version info from GO data."""
        go_data = {
            "graphs": [{
                "id": "http://purl.obolibrary.org/obo/go.owl",
                "meta": {
                    "version": "http://purl.obolibrary.org/obo/go/releases/2025-03-16/go.owl",
                    "basicPropertyValues": [
                        {
                            "pred": "http://www.w3.org/2002/07/owl#versionInfo", 
                            "val": "2025-03-16"
                        }
                    ]
                },
                "nodes": []
            }]
        }
        
        version_manager = OntologyVersionManager()
        version_info = version_manager.extract_go_version_info(go_data)
        
        assert version_info["version_date"] == "2025-03-16"
        assert version_info["version_url"] == "http://purl.obolibrary.org/obo/go/releases/2025-03-16/go.owl"

    def test_version_manager_compare_versions_no_stored_data(self):
        """Test version comparison when no stored data exists."""
        go_data = {
            "graphs": [{
                "id": "http://purl.obolibrary.org/obo/go.owl",
                "meta": {
                    "version": "http://purl.obolibrary.org/obo/go/releases/2025-03-16/go.owl",
                    "basicPropertyValues": [
                        {
                            "pred": "http://www.w3.org/2002/07/owl#versionInfo", 
                            "val": "2025-03-16"
                        }
                    ]
                },
                "nodes": []
            }]
        }
        
        version_manager = OntologyVersionManager()
        
        with patch.object(version_manager, 'get_stored_version_info', return_value=None):
            needs_update, stored_info, new_version_info = version_manager.compare_versions("GO", go_data)
        
        assert needs_update is True
        assert stored_info is None
        assert new_version_info["version_date"] == "2025-03-16"

    def test_version_manager_compare_versions_same_version(self):
        """Test version comparison when versions are the same."""
        go_data = {
            "graphs": [{
                "id": "http://purl.obolibrary.org/obo/go.owl",
                "meta": {
                    "version": "http://purl.obolibrary.org/obo/go/releases/2025-03-16/go.owl",
                    "basicPropertyValues": [
                        {
                            "pred": "http://www.w3.org/2002/07/owl#versionInfo", 
                            "val": "2025-03-16"
                        }
                    ]
                },
                "nodes": [{"id": "GO:0001", "lbl": "test"}]
            }]
        }
        
        stored_data = {
            "version_info": {"version_date": "2025-03-16", "content_hash": "abc123"},
            "collection_name": "GO_existing"
        }
        
        version_manager = OntologyVersionManager()
        
        with patch.object(version_manager, 'get_stored_version_info', return_value=stored_data):
            with patch.object(version_manager, 'generate_version_hash', return_value="abc123"):
                needs_update, stored_info, new_version_info = version_manager.compare_versions("GO", go_data)
        
        assert needs_update is False
        assert stored_info == stored_data
        assert new_version_info["version_date"] == "2025-03-16"

    def test_version_manager_compare_versions_different_content(self):
        """Test version comparison when content has changed."""
        go_data = {
            "meta": {"version": "2025-03-16"},
            "graphs": [{"nodes": [{"id": "GO:0001", "lbl": "updated test"}]}]
        }
        
        stored_data = {
            "version_info": {"version_date": "2025-03-16", "content_hash": "abc123"},
            "collection_name": "GO_existing"
        }
        
        version_manager = OntologyVersionManager()
        
        with patch.object(version_manager, 'get_stored_version_info', return_value=stored_data):
            with patch.object(version_manager, 'generate_version_hash', return_value="def456"):
                needs_update, stored_info, new_version_info = version_manager.compare_versions("GO", go_data)
        
        assert needs_update is True
        assert stored_info == stored_data
        assert new_version_info["content_hash"] == "def456"

    def test_version_manager_metadata_path(self):
        """Test metadata path generation."""
        version_manager = OntologyVersionManager()
        path = version_manager.get_version_metadata_path("GO")
        assert path.endswith("GO_version_metadata.json")

    def test_version_manager_file_not_found(self):
        """Test behavior when metadata file doesn't exist."""
        version_manager = OntologyVersionManager()
        
        # Use a non-existent ontology name
        result = version_manager.get_stored_version_info("NONEXISTENT_ONTOLOGY")
        assert result is None