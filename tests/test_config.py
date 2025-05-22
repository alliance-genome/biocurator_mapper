import os
import tempfile
import pytest
import yaml
from unittest.mock import patch, mock_open

from app.config import load_ontology_config, ONTOLOGY_CONFIG


class TestOntologyConfig:
    """Test ontology configuration loading."""

    def test_load_ontology_config_file_exists(self):
        """Test loading config when file exists."""
        mock_config = {
            "ontologies": {
                "GO": {"name": "Gene Ontology", "enabled": True},
                "DOID": {"name": "Disease Ontology", "enabled": True}
            },
            "settings": {"default_k": 5}
        }
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_config))):
            with patch("os.path.exists", return_value=True):
                config = load_ontology_config()
                
        assert config["ontologies"]["GO"]["name"] == "Gene Ontology"
        assert config["ontologies"]["DOID"]["name"] == "Disease Ontology"
        assert config["settings"]["default_k"] == 5

    def test_load_ontology_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            config = load_ontology_config()
            
        assert "ontologies" in config
        assert "GO" in config["ontologies"]
        assert "DOID" in config["ontologies"]
        assert config["ontologies"]["GO"]["enabled"] is True

    def test_ontology_config_loaded_on_import(self):
        """Test that ONTOLOGY_CONFIG is loaded on module import."""
        assert ONTOLOGY_CONFIG is not None
        assert "ontologies" in ONTOLOGY_CONFIG

    def test_ontology_config_environment_variable(self):
        """Test that ONTOLOGY_CONFIG_PATH environment variable is respected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {
                "ontologies": {"TEST": {"name": "Test Ontology", "enabled": True}},
                "settings": {"default_k": 10}
            }
            yaml.dump(test_config, f)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"ONTOLOGY_CONFIG_PATH": temp_path}):
                from importlib import reload
                import app.config
                reload(app.config)
                
                config = app.config.load_ontology_config()
                assert "TEST" in config["ontologies"]
                assert config["settings"]["default_k"] == 10
        finally:
            os.unlink(temp_path)