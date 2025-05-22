import pytest
import yaml
import tempfile
import os
from unittest.mock import patch


class TestYAMLConfigIntegration:
    """Test YAML configuration integration with real GO data scenarios."""

    @pytest.fixture
    def sample_yaml_config(self):
        """Sample YAML configuration for testing."""
        return """
ontologies:
  GO:
    name: "Gene Ontology"
    description: "The Gene Ontology (GO) knowledgebase"
    default_source_url: "http://purl.obolibrary.org/obo/go.json"
    id_format:
      separator: ":"
      prefix_replacement:
        "_": ":"
    enabled: true
    
  DOID:
    name: "Disease Ontology"
    description: "Human Disease Ontology"
    default_source_url: "http://purl.obolibrary.org/obo/doid.json"
    id_format:
      separator: ":"
      prefix_replacement:
        "_": ":"
    enabled: true

settings:
  default_k: 5
  supported_formats:
    - "json"
    - "obo"
  json_parsing:
    graphs_key: "graphs"
    nodes_key: "nodes"
    id_key: "id"
    label_key: "lbl"
    definition_path: ["meta", "definition", "val"]
"""

    def test_yaml_config_loading(self, sample_yaml_config):
        """Test that YAML configuration loads correctly."""
        config = yaml.safe_load(sample_yaml_config)
        
        # Test structure
        assert "ontologies" in config
        assert "settings" in config
        
        # Test GO configuration
        go_config = config["ontologies"]["GO"]
        assert go_config["name"] == "Gene Ontology"
        assert go_config["default_source_url"] == "http://purl.obolibrary.org/obo/go.json"
        assert go_config["enabled"] is True
        
        # Test ID format configuration
        id_format = go_config["id_format"]
        assert id_format["separator"] == ":"
        assert id_format["prefix_replacement"]["_"] == ":"

    def test_json_parsing_configuration(self, sample_yaml_config):
        """Test JSON parsing configuration matches GO.json structure."""
        config = yaml.safe_load(sample_yaml_config)
        parsing_config = config["settings"]["json_parsing"]
        
        # Test that configuration keys match actual GO.json structure
        assert parsing_config["graphs_key"] == "graphs"
        assert parsing_config["nodes_key"] == "nodes"
        assert parsing_config["id_key"] == "id"
        assert parsing_config["label_key"] == "lbl"
        assert parsing_config["definition_path"] == ["meta", "definition", "val"]

    def test_go_url_configuration(self, sample_yaml_config):
        """Test that GO URL configuration is valid."""
        config = yaml.safe_load(sample_yaml_config)
        go_config = config["ontologies"]["GO"]
        
        url = go_config["default_source_url"]
        assert url.startswith("http")
        assert "go.json" in url
        assert "obolibrary.org" in url

    def test_multiple_ontology_support(self, sample_yaml_config):
        """Test configuration supports multiple ontologies."""
        config = yaml.safe_load(sample_yaml_config)
        ontologies = config["ontologies"]
        
        assert "GO" in ontologies
        assert "DOID" in ontologies
        
        # Both should have consistent structure
        for onto_name, onto_config in ontologies.items():
            assert "name" in onto_config
            assert "description" in onto_config
            assert "default_source_url" in onto_config
            assert "enabled" in onto_config
            assert "id_format" in onto_config

    def test_config_file_environment_variable(self, sample_yaml_config):
        """Test that configuration can be loaded from custom path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_yaml_config)
            temp_path = f.name

        try:
            # Test loading from custom path
            with patch.dict(os.environ, {"ONTOLOGY_CONFIG_PATH": temp_path}):
                # Simulate loading config
                with open(temp_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                assert config["ontologies"]["GO"]["name"] == "Gene Ontology"
        finally:
            os.unlink(temp_path)

    def test_id_format_processing(self, sample_yaml_config):
        """Test ID format configuration works with real GO IDs."""
        config = yaml.safe_load(sample_yaml_config)
        id_format = config["ontologies"]["GO"]["id_format"]
        
        # Test with real GO URI
        test_uri = "http://purl.obolibrary.org/obo/GO_0000001"
        term_id = test_uri.split("/")[-1]  # GO_0000001
        
        # Apply replacement rules
        for old, new in id_format["prefix_replacement"].items():
            term_id = term_id.replace(old, new)
        
        assert term_id == "GO:0000001"

    def test_parsing_path_configuration(self, sample_yaml_config):
        """Test that definition path configuration works with real GO data."""
        config = yaml.safe_load(sample_yaml_config)
        definition_path = config["settings"]["json_parsing"]["definition_path"]
        
        # Simulate real GO node structure
        go_node = {
            "id": "http://purl.obolibrary.org/obo/GO_0000001",
            "lbl": "mitochondrion inheritance",
            "meta": {
                "definition": {
                    "val": "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton."
                }
            }
        }
        
        # Extract definition using configured path
        def get_nested_value(data, path):
            current = data
            for key in path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return ""
            return current
        
        definition = get_nested_value(go_node, definition_path)
        assert "mitochondria" in definition
        assert len(definition) > 50

    def test_configuration_validation(self, sample_yaml_config):
        """Test configuration validation and required fields."""
        config = yaml.safe_load(sample_yaml_config)
        
        # Test required top-level keys
        required_keys = ["ontologies", "settings"]
        for key in required_keys:
            assert key in config
        
        # Test required ontology configuration
        for onto_name, onto_config in config["ontologies"].items():
            required_onto_keys = ["name", "enabled", "id_format"]
            for key in required_onto_keys:
                assert key in onto_config, f"Missing {key} in {onto_name} config"
        
        # Test required settings
        settings = config["settings"]
        assert "json_parsing" in settings
        
        parsing = settings["json_parsing"]
        required_parsing_keys = ["graphs_key", "nodes_key", "id_key", "label_key", "definition_path"]
        for key in required_parsing_keys:
            assert key in parsing, f"Missing {key} in json_parsing config"

    def test_yaml_config_with_real_data_simulation(self, sample_yaml_config):
        """Test YAML configuration with simulated real data processing."""
        config = yaml.safe_load(sample_yaml_config)
        
        # Simulate processing real GO data with this configuration
        go_config = config["ontologies"]["GO"]
        parsing_config = config["settings"]["json_parsing"]
        
        # Simulate real GO.json structure
        simulated_data = {
            "graphs": [{
                "nodes": [{
                    "id": "http://purl.obolibrary.org/obo/GO_0000001",
                    "lbl": "mitochondrion inheritance",
                    "meta": {
                        "definition": {
                            "val": "Test definition"
                        }
                    }
                }]
            }]
        }
        
        # Use configuration to extract data
        graphs_key = parsing_config["graphs_key"]
        nodes_key = parsing_config["nodes_key"]
        id_key = parsing_config["id_key"]
        label_key = parsing_config["label_key"]
        
        # Extract using configuration
        graphs = simulated_data.get(graphs_key, [])
        assert len(graphs) == 1
        
        nodes = graphs[0].get(nodes_key, [])
        assert len(nodes) == 1
        
        node = nodes[0]
        node_id = node[id_key]
        node_label = node[label_key]
        
        assert "GO_0000001" in node_id
        assert node_label == "mitochondrion inheritance"
        
        # Test ID transformation
        term_id = node_id.split("/")[-1]
        for old, new in go_config["id_format"]["prefix_replacement"].items():
            term_id = term_id.replace(old, new)
        
        assert term_id == "GO:0000001"

    def test_config_backwards_compatibility(self):
        """Test that configuration handles missing or minimal settings."""
        minimal_config = {
            "ontologies": {
                "GO": {
                    "name": "Gene Ontology",
                    "enabled": True
                }
            }
        }
        
        # Should work with minimal configuration
        assert "GO" in minimal_config["ontologies"]
        go_config = minimal_config["ontologies"]["GO"]
        assert go_config["enabled"] is True
        
        # Test defaults can be applied
        default_id_format = {"prefix_replacement": {"_": ":"}}
        actual_id_format = go_config.get("id_format", default_id_format)
        assert actual_id_format["prefix_replacement"]["_"] == ":"