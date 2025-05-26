"""Test core DO parsing functionality for integration tests."""
import json
import pytest
from pathlib import Path

from app.go_parser import parse_enhanced_go_term, parse_go_json_enhanced
from app.ontology_manager import OntologyManager


class TestDOParserCoreFunctionality:
    """Test core DO parsing functionality with real DO data structures."""

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

    def test_parse_do_node_basic_structure(self, sample_do_data):
        """Test parsing of basic DO node structure."""
        # Get first node from test data
        nodes = sample_do_data["graphs"][0]["nodes"]
        node = nodes[0]  # angiosarcoma node
        
        # Parse using go_parser (which handles DO format)
        result = parse_enhanced_go_term(node)
        
        assert result is not None
        assert result["id"] == "DOID:0001816"
        assert result["name"] == "angiosarcoma"
        assert "malignant vascular tumor" in result["definition"]

    def test_parse_do_synonym_extraction(self, sample_do_data):
        """Test extraction of different DO synonym types."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Find node with multiple synonym types (angiosarcoma)
        angiosarcoma_node = nodes[0]
        result = parse_enhanced_go_term(angiosarcoma_node)
        
        # Test exact synonyms
        assert "hemangiosarcoma" in result["exact_synonyms"]
        
        # Test narrow synonyms
        assert "epithelioid angiosarcoma" in result["narrow_synonyms"]
        
        # Test related synonyms
        assert "malignant hemangioendothelioma" in result["related_synonyms"]
        
        # Test all_synonyms contains all types
        assert len(result["all_synonyms"]) >= 3
        assert "hemangiosarcoma" in result["all_synonyms"]
        assert "epithelioid angiosarcoma" in result["all_synonyms"]
        assert "malignant hemangioendothelioma" in result["all_synonyms"]

    def test_parse_do_complex_synonyms(self, sample_do_data):
        """Test parsing of complex synonym structures in DO."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Find diabetes node with many synonyms
        diabetes_node = next(node for node in nodes if "diabetes mellitus" in node["lbl"])
        result = parse_enhanced_go_term(diabetes_node)
        
        # Should have multiple exact synonyms
        assert len(result["exact_synonyms"]) >= 3
        assert "Type II diabetes mellitus" in result["exact_synonyms"]
        assert "diabetes mellitus type 2" in result["exact_synonyms"]
        assert "NIDDM" in result["exact_synonyms"]
        
        # Should have related synonyms
        assert "adult-onset diabetes" in result["related_synonyms"]
        
        # Should have narrow synonyms
        assert "maturity-onset diabetes of the young" in result["narrow_synonyms"]

    def test_parse_do_cross_references(self, sample_do_data):
        """Test extraction of cross-references from DO nodes."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Test node with xrefs (type 2 diabetes)
        diabetes_node = next(node for node in nodes if "type 2 diabetes" in node["lbl"])
        result = parse_enhanced_go_term(diabetes_node)
        
        # Should extract cross-references
        assert len(result["cross_references"]) > 0
        
        # Check for specific cross-references that should be in the test data
        xref_vals = result["cross_references"]
        # Should contain MESH, ICD, and other database references
        mesh_refs = [xref for xref in xref_vals if "MESH:" in str(xref)]
        icd_refs = [xref for xref in xref_vals if "ICD" in str(xref)]
        
        assert len(mesh_refs) > 0, f"No MESH references found in: {xref_vals}"
        assert len(icd_refs) > 0, f"No ICD references found in: {xref_vals}"

    def test_parse_do_namespace_extraction(self, sample_do_data):
        """Test extraction of DO namespace information."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        for node in nodes[:3]:  # Test first few nodes
            result = parse_enhanced_go_term(node)
            
            # All DO terms should have disease_ontology namespace
            assert result["namespace"] == "disease_ontology"

    def test_parse_do_minimal_node(self):
        """Test parsing of minimal DO node with required fields only."""
        minimal_node = {
            "id": "http://purl.obolibrary.org/obo/DOID_0000001",
            "lbl": "disease",
            "type": "CLASS",
            "meta": {
                "basicPropertyValues": [
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
                        "val": "disease_ontology"
                    }
                ]
            }
        }
        
        result = parse_enhanced_go_term(minimal_node)
        
        assert result is not None
        assert result["id"] == "DOID:0000001"
        assert result["name"] == "disease"
        assert result["definition"] == ""  # No definition provided
        assert result["namespace"] == "disease_ontology"
        assert result["exact_synonyms"] == []
        assert result["all_synonyms"] == []

    def test_parse_do_invalid_node(self):
        """Test handling of invalid DO node structures."""
        # Missing required fields
        invalid_node = {
            "type": "CLASS",
            "meta": {}
            # Missing 'id' and 'lbl'
        }
        
        result = parse_enhanced_go_term(invalid_node)
        assert result is None

    def test_parse_full_do_json_structure(self, sample_do_data):
        """Test parsing of complete DO JSON structure."""
        # Parse entire DO JSON using the main parsing function
        parsed_terms = parse_go_json_enhanced(sample_do_data)
        
        # Should return list of parsed terms
        assert isinstance(parsed_terms, list)
        assert len(parsed_terms) > 0
        
        # Each term should have required structure
        for term in parsed_terms:
            assert "id" in term
            assert "name" in term
            assert "definition" in term
            assert "exact_synonyms" in term
            assert "narrow_synonyms" in term
            assert "broad_synonyms" in term
            assert "all_synonyms" in term
            assert "cross_references" in term
            assert "namespace" in term
            assert "searchable_text" in term

    def test_ontology_manager_do_integration(self, sample_do_data, ontology_manager):
        """Test OntologyManager integration with DO parsing."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Test with raw DO node
        raw_node = nodes[0]
        result = ontology_manager._extract_enhanced_term_data(raw_node)
        
        # Should properly extract data using DO parser
        assert result["term_id"] == "DOID:0001816"
        assert result["name"] == "angiosarcoma"
        assert len(result["exact_synonyms"]) > 0
        assert "hemangiosarcoma" in result["exact_synonyms"]

    def test_do_searchable_text_generation(self, sample_do_data):
        """Test generation of searchable text for DO terms."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        covid_node = next(node for node in nodes if "COVID-19" in node["lbl"])
        
        result = parse_enhanced_go_term(covid_node)
        
        # Searchable text should include name, definition, and synonyms
        searchable = result["searchable_text"]
        
        assert "COVID-19" in searchable
        assert "coronavirus disease 2019" in searchable
        assert "SARS-CoV-2 infection" in searchable
        assert "respiratory tract" in searchable  # From definition

    def test_do_edge_cases_handling(self):
        """Test handling of DO-specific edge cases."""
        # Node with no synonyms
        no_synonyms_node = {
            "id": "http://purl.obolibrary.org/obo/DOID_0000001",
            "lbl": "disease",
            "type": "CLASS",
            "meta": {
                "definition": {
                    "val": "A disposition to undergo pathological processes.",
                    "xrefs": []
                },
                "basicPropertyValues": [
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
                        "val": "disease_ontology"
                    }
                ]
            }
        }
        
        result = parse_enhanced_go_term(no_synonyms_node)
        
        assert result is not None
        assert result["exact_synonyms"] == []
        assert result["all_synonyms"] == []
        assert len(result["definition"]) > 0

    def test_do_unicode_handling(self):
        """Test handling of unicode characters in DO terms."""
        unicode_node = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease with ñ and é characters",
            "type": "CLASS",
            "meta": {
                "definition": {
                    "val": "A disease with unicode characters: α, β, γ"
                },
                "synonyms": [
                    {
                        "pred": "hasExactSynonym",
                        "val": "synonym with ü and ø"
                    }
                ],
                "basicPropertyValues": [
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
                        "val": "disease_ontology"
                    }
                ]
            }
        }
        
        result = parse_enhanced_go_term(unicode_node)
        
        assert result is not None
        assert "ñ and é" in result["name"]
        assert "α, β, γ" in result["definition"]
        assert "ü and ø" in result["exact_synonyms"][0]

    def test_do_performance_parsing(self, sample_do_data):
        """Test parsing performance with realistic data sizes."""
        import time
        
        start_time = time.time()
        parsed_terms = parse_go_json_enhanced(sample_do_data)
        end_time = time.time()
        
        parsing_time = end_time - start_time
        
        # Should parse reasonably quickly (less than 1 second for test data)
        assert parsing_time < 1.0, f"Parsing took too long: {parsing_time:.2f} seconds"
        assert len(parsed_terms) > 0
        
        # Each term should be properly parsed
        for term in parsed_terms:
            assert term["id"].startswith("DOID:")
            assert len(term["name"]) > 0