"""Test OntologyManager integration with DO parser for complete data flow."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.ontology_manager import OntologyManager
from app.go_parser import parse_go_json_enhanced


class TestDOOntologyManagerIntegration:
    """Test complete integration of DO parsing with OntologyManager."""

    @pytest.fixture
    def sample_do_data(self):
        """Load comprehensive DO test data."""
        test_data_path = Path(__file__).parent / "data" / "sample_do_comprehensive.json"
        with open(test_data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def ontology_manager(self):
        """Create OntologyManager instance for testing."""
        return OntologyManager()

    @pytest.fixture
    def mock_weaviate_client(self):
        """Mock Weaviate client for testing."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        mock_client.collections.create.return_value = None
        mock_client.collections.get.return_value.data.insert_many.return_value = MagicMock()
        return mock_client

    def test_extract_enhanced_term_data_with_real_do_nodes(self, ontology_manager, sample_do_data):
        """Test _extract_enhanced_term_data with actual DO node structures."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        for node in nodes:
            result = ontology_manager._extract_enhanced_term_data(node)
            
            # Verify required fields are present
            assert "term_id" in result
            assert "name" in result
            assert "definition" in result
            assert "exact_synonyms" in result
            assert "narrow_synonyms" in result
            assert "broad_synonyms" in result
            assert "all_synonyms" in result
            assert "searchable_text" in result
            
            # Verify ID format transformation
            assert result["term_id"].startswith("DOID:")
            
            # Verify name is preserved
            assert result["name"] == node["lbl"]
            
            # Verify synonym categorization
            meta = node.get("meta", {})
            synonyms = meta.get("synonyms", [])
            
            exact_count = len([s for s in synonyms if s.get("pred") == "hasExactSynonym"])
            narrow_count = len([s for s in synonyms if s.get("pred") == "hasNarrowSynonym"])
            broad_count = len([s for s in synonyms if s.get("pred") == "hasBroadSynonym"])
            related_count = len([s for s in synonyms if s.get("pred") == "hasRelatedSynonym"])
            
            assert len(result["exact_synonyms"]) == exact_count
            assert len(result["narrow_synonyms"]) == narrow_count
            assert len(result["broad_synonyms"]) == broad_count
            # all_synonyms should contain all types
            assert len(result["all_synonyms"]) == exact_count + narrow_count + broad_count + related_count

    def test_build_searchable_text_integration(self, ontology_manager, sample_do_data):
        """Test searchable text generation with real DO data."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Test with COVID-19 node (has good variety of content)
        covid_node = next(node for node in nodes if "COVID-19" in node["lbl"])
        term_data = ontology_manager._extract_enhanced_term_data(covid_node)
        
        searchable_text = ontology_manager._build_searchable_text(term_data)
        
        # Should contain name
        assert "COVID-19" in searchable_text
        
        # Should contain definition content
        assert "respiratory tract" in searchable_text or "infection" in searchable_text
        
        # Should contain synonyms
        assert "coronavirus disease 2019" in searchable_text
        assert "SARS-CoV-2 infection" in searchable_text

    def test_build_searchable_text_with_preprocessing(self, ontology_manager):
        """Test searchable text generation with different preprocessing options."""
        # Mock different configurations
        term_data = {
            "name": "Type 2 Diabetes",
            "definition": "A metabolic disorder, characterized by high blood glucose.",
            "exact_synonyms": ["T2DM", "Non-insulin Dependent Diabetes"],
            "narrow_synonyms": ["Adult-Onset Diabetes"],
            "broad_synonyms": ["Diabetes Mellitus"]
        }
        
        # Test with default configuration
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
            "preprocessing": {"combine_fields_separator": " | "}
        }):
            searchable_text = ontology_manager._build_searchable_text(term_data)
            assert "Type 2 Diabetes" in searchable_text
            assert "metabolic disorder" in searchable_text
            assert "T2DM" in searchable_text
            assert " | " in searchable_text

    def test_extract_enhanced_term_data_error_handling(self, ontology_manager):
        """Test error handling in _extract_enhanced_term_data."""
        # Test with malformed node
        malformed_node = {
            "id": "malformed",
            # Missing 'lbl' field
            "meta": {}
        }
        
        # Should not crash, should return fallback data
        result = ontology_manager._extract_enhanced_term_data(malformed_node)
        assert result["term_id"] == "malformed"
        assert result["name"] == ""
        assert result["exact_synonyms"] == []

    def test_extract_enhanced_term_data_with_empty_meta(self, ontology_manager):
        """Test extraction with node that has empty meta."""
        empty_meta_node = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {}
        }
        
        result = ontology_manager._extract_enhanced_term_data(empty_meta_node)
        
        assert result["term_id"] == "DOID:12345"
        assert result["name"] == "test disease"
        assert result["definition"] == ""
        assert result["exact_synonyms"] == []
        assert result["all_synonyms"] == []

    def test_parse_full_do_json_to_enhanced_terms(self, sample_do_data):
        """Test parsing full DO JSON to enhanced terms format."""
        # Parse using the go_parser which handles DO format
        parsed_terms = parse_go_json_enhanced(sample_do_data)
        
        assert len(parsed_terms) > 0
        
        # Check each term has the enhanced structure
        for term in parsed_terms:
            assert "id" in term and term["id"].startswith("DOID:")
            assert "name" in term and len(term["name"]) > 0
            assert "definition" in term
            assert "exact_synonyms" in term
            assert "narrow_synonyms" in term
            assert "broad_synonyms" in term
            assert "all_synonyms" in term
            assert "cross_references" in term
            assert "namespace" in term
            assert "searchable_text" in term

    def test_ontology_manager_with_parsed_do_terms(self, ontology_manager, sample_do_data):
        """Test OntologyManager processing of pre-parsed DO terms."""
        # First parse using go_parser
        parsed_terms = parse_go_json_enhanced(sample_do_data)
        
        # Then process each through OntologyManager
        for parsed_term in parsed_terms[:3]:  # Test first few
            result = ontology_manager._extract_enhanced_term_data(parsed_term)
            
            # Should preserve the parsed structure
            assert result["term_id"] == parsed_term["id"]
            assert result["name"] == parsed_term["name"]
            assert result["definition"] == parsed_term["definition"]
            assert result["exact_synonyms"] == parsed_term["exact_synonyms"]
            assert result["narrow_synonyms"] == parsed_term["narrow_synonyms"]
            assert result["broad_synonyms"] == parsed_term["broad_synonyms"]
            assert result["all_synonyms"] == parsed_term["all_synonyms"]

    @pytest.mark.asyncio
    async def test_weaviate_integration_preparation(self, ontology_manager, sample_do_data, mock_weaviate_client):
        """Test preparation of DO data for Weaviate integration."""
        # Parse DO data
        parsed_terms = parse_go_json_enhanced(sample_do_data)
        
        # Process terms through OntologyManager
        enhanced_terms = []
        for term in parsed_terms:
            enhanced_data = ontology_manager._extract_enhanced_term_data(term)
            enhanced_data["searchable_text"] = ontology_manager._build_searchable_text(enhanced_data)
            enhanced_terms.append(enhanced_data)
        
        # Verify data is ready for Weaviate
        for term in enhanced_terms:
            # Check required fields for Weaviate
            assert "term_id" in term
            assert "name" in term
            assert "searchable_text" in term
            assert len(term["searchable_text"]) > 0
            
            # Check data types
            assert isinstance(term["exact_synonyms"], list)
            assert isinstance(term["narrow_synonyms"], list)
            assert isinstance(term["broad_synonyms"], list)
            assert isinstance(term["all_synonyms"], list)

    def test_do_data_consistency_across_processing(self, ontology_manager, sample_do_data):
        """Test that DO data remains consistent through the processing pipeline."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Pick a complex node for testing
        diabetes_node = next(node for node in nodes if "type 2 diabetes" in node["lbl"])
        original_name = diabetes_node["lbl"]
        
        # Process through OntologyManager
        enhanced_data = ontology_manager._extract_enhanced_term_data(diabetes_node)
        
        # Verify consistency
        assert enhanced_data["name"] == original_name
        assert enhanced_data["term_id"] == "DOID:9352"  # Known ID for type 2 diabetes
        
        # Build searchable text
        searchable_text = ontology_manager._build_searchable_text(enhanced_data)
        enhanced_data["searchable_text"] = searchable_text
        
        # Verify searchable text contains key information
        assert original_name in searchable_text
        assert len(enhanced_data["exact_synonyms"]) >= 3  # Should have multiple exact synonyms

    def test_performance_with_realistic_do_data_size(self, ontology_manager, sample_do_data):
        """Test performance of OntologyManager with realistic DO data sizes."""
        import time
        
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        start_time = time.time()
        
        # Process all nodes
        processed_terms = []
        for node in nodes:
            enhanced_data = ontology_manager._extract_enhanced_term_data(node)
            searchable_text = ontology_manager._build_searchable_text(enhanced_data)
            enhanced_data["searchable_text"] = searchable_text
            processed_terms.append(enhanced_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        assert processing_time < 2.0, f"Processing took too long: {processing_time:.2f} seconds"
        assert len(processed_terms) == len(nodes)
        
        # Verify all terms were processed correctly
        for term in processed_terms:
            assert len(term["term_id"]) > 0
            assert len(term["name"]) > 0
            assert len(term["searchable_text"]) > 0

    def test_configuration_impact_on_processing(self, ontology_manager):
        """Test how different configurations affect DO term processing."""
        test_term_data = {
            "name": "Lung Cancer",
            "definition": "A respiratory system cancer located in the lung.",
            "exact_synonyms": ["cancer of lung", "pulmonary cancer"],
            "narrow_synonyms": ["adenocarcinoma of lung"],
            "broad_synonyms": ["respiratory cancer"]
        }
        
        # Test with name only
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "vectorize_fields": {"name": True, "definition": False, "synonyms": False},
            "preprocessing": {"combine_fields_separator": " "}
        }):
            searchable_text = ontology_manager._build_searchable_text(test_term_data)
            assert "Lung Cancer" in searchable_text
            assert "respiratory system cancer" not in searchable_text
            assert "cancer of lung" not in searchable_text
        
        # Test with all fields
        with patch('app.ontology_manager.EMBEDDINGS_CONFIG', {
            "vectorize_fields": {"name": True, "definition": True, "synonyms": True},
            "preprocessing": {"combine_fields_separator": " "}
        }):
            searchable_text = ontology_manager._build_searchable_text(test_term_data)
            assert "Lung Cancer" in searchable_text
            assert "respiratory system cancer" in searchable_text
            assert "cancer of lung" in searchable_text
            assert "adenocarcinoma of lung" in searchable_text