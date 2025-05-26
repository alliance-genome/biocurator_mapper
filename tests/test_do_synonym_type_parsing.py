"""Test DO synonym type parsing: exact, narrow, broad, and related synonyms."""
import json
import pytest
from pathlib import Path

from app.go_parser import extract_synonyms_from_go_node, parse_enhanced_go_term
from app.ontology_manager import OntologyManager


class TestDOSynonymTypeParsing:
    """Test comprehensive synonym type parsing for DO terms."""

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
    def angiosarcoma_node(self, sample_do_data):
        """Get the angiosarcoma node which has multiple synonym types."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "angiosarcoma" in node["lbl"])

    @pytest.fixture
    def diabetes_node(self, sample_do_data):
        """Get the type 2 diabetes node which has many synonym variations."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "type 2 diabetes" in node["lbl"])

    @pytest.fixture
    def covid_node(self, sample_do_data):
        """Get the COVID-19 node which has official naming synonyms."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "COVID-19" in node["lbl"])

    def test_exact_synonym_extraction(self, angiosarcoma_node):
        """Test extraction of hasExactSynonym type synonyms."""
        synonym_data = extract_synonyms_from_go_node(angiosarcoma_node)
        
        # Should have exact synonyms
        assert len(synonym_data["exact_synonyms"]) > 0
        assert "hemangiosarcoma" in synonym_data["exact_synonyms"]
        
        # Exact synonyms should be in all_synonyms
        for exact_syn in synonym_data["exact_synonyms"]:
            assert exact_syn in synonym_data["all_synonyms"]

    def test_narrow_synonym_extraction(self, angiosarcoma_node):
        """Test extraction of hasNarrowSynonym type synonyms."""
        synonym_data = extract_synonyms_from_go_node(angiosarcoma_node)
        
        # Should have narrow synonyms
        assert len(synonym_data["narrow_synonyms"]) > 0
        assert "epithelioid angiosarcoma" in synonym_data["narrow_synonyms"]
        
        # Narrow synonyms should be in all_synonyms
        for narrow_syn in synonym_data["narrow_synonyms"]:
            assert narrow_syn in synonym_data["all_synonyms"]

    def test_related_synonym_extraction(self, angiosarcoma_node):
        """Test extraction of hasRelatedSynonym type synonyms."""
        synonym_data = extract_synonyms_from_go_node(angiosarcoma_node)
        
        # Should have related synonyms
        assert len(synonym_data["related_synonyms"]) > 0
        assert "malignant hemangioendothelioma" in synonym_data["related_synonyms"]
        
        # Related synonyms should be in all_synonyms
        for related_syn in synonym_data["related_synonyms"]:
            assert related_syn in synonym_data["all_synonyms"]

    def test_broad_synonym_extraction(self, sample_do_data):
        """Test extraction of hasBroadSynonym type synonyms."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        # Find nodes with broad synonyms
        nodes_with_broad = []
        for node in nodes:
            synonym_data = extract_synonyms_from_go_node(node)
            if synonym_data["broad_synonyms"]:
                nodes_with_broad.append(node)
        
        assert len(nodes_with_broad) > 0, "Should find nodes with broad synonyms"
        
        # Test the first node with broad synonyms
        test_node = nodes_with_broad[0]
        synonym_data = extract_synonyms_from_go_node(test_node)
        
        # Broad synonyms should be in all_synonyms
        for broad_syn in synonym_data["broad_synonyms"]:
            assert broad_syn in synonym_data["all_synonyms"]

    def test_diabetes_comprehensive_synonyms(self, diabetes_node):
        """Test comprehensive synonym extraction for diabetes with many synonym types."""
        result = parse_enhanced_go_term(diabetes_node)
        
        # Should have multiple exact synonyms
        assert len(result["exact_synonyms"]) >= 3
        expected_exact = ["Type II diabetes mellitus", "diabetes mellitus type 2", "NIDDM"]
        for expected in expected_exact:
            assert expected in result["exact_synonyms"], f"Missing exact synonym: {expected}"
        
        # Should have related synonyms
        assert len(result["related_synonyms"]) >= 1
        assert "adult-onset diabetes" in result["related_synonyms"]
        
        # Should have narrow synonyms
        assert len(result["narrow_synonyms"]) >= 1
        assert "maturity-onset diabetes of the young" in result["narrow_synonyms"]
        
        # All synonyms should contain all types
        total_synonyms = (len(result["exact_synonyms"]) + 
                         len(result["narrow_synonyms"]) + 
                         len(result["broad_synonyms"]) + 
                         len(result["related_synonyms"]))
        assert len(result["all_synonyms"]) == total_synonyms

    def test_covid_exact_synonyms(self, covid_node):
        """Test exact synonym extraction for COVID-19 official names."""
        result = parse_enhanced_go_term(covid_node)
        
        # Should have official exact synonyms
        expected_exact = [
            "coronavirus disease 2019",
            "2019-nCoV disease", 
            "SARS-CoV-2 infection"
        ]
        
        for expected in expected_exact:
            assert expected in result["exact_synonyms"], f"Missing COVID exact synonym: {expected}"
        
        # Should have related synonym
        assert "novel coronavirus pneumonia" in result["related_synonyms"]

    def test_synonym_type_categorization_accuracy(self, sample_do_data):
        """Test that synonym types are categorized correctly across all nodes."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        for node in nodes:
            if "meta" not in node or "synonyms" not in node["meta"]:
                continue
                
            # Get raw synonyms from node
            raw_synonyms = node["meta"]["synonyms"]
            
            # Parse using our function
            synonym_data = extract_synonyms_from_go_node(node)
            
            # Check each raw synonym is categorized correctly
            for raw_syn in raw_synonyms:
                syn_text = raw_syn.get("val", "")
                syn_type = raw_syn.get("pred", "")
                
                if syn_type == "hasExactSynonym":
                    assert syn_text in synonym_data["exact_synonyms"], \
                        f"Exact synonym '{syn_text}' not in exact_synonyms for {node['lbl']}"
                elif syn_type == "hasNarrowSynonym":
                    assert syn_text in synonym_data["narrow_synonyms"], \
                        f"Narrow synonym '{syn_text}' not in narrow_synonyms for {node['lbl']}"
                elif syn_type == "hasBroadSynonym":
                    assert syn_text in synonym_data["broad_synonyms"], \
                        f"Broad synonym '{syn_text}' not in broad_synonyms for {node['lbl']}"
                elif syn_type == "hasRelatedSynonym":
                    assert syn_text in synonym_data["related_synonyms"], \
                        f"Related synonym '{syn_text}' not in related_synonyms for {node['lbl']}"
                
                # All should be in all_synonyms
                assert syn_text in synonym_data["all_synonyms"], \
                    f"Synonym '{syn_text}' not in all_synonyms for {node['lbl']}"

    def test_empty_synonym_handling(self):
        """Test handling of nodes with no synonyms."""
        node_no_synonyms = {
            "id": "http://purl.obolibrary.org/obo/DOID_0000001",
            "lbl": "disease",
            "meta": {
                "definition": {
                    "val": "A disposition to undergo pathological processes."
                },
                "synonyms": []  # Empty synonyms
            }
        }
        
        synonym_data = extract_synonyms_from_go_node(node_no_synonyms)
        
        assert synonym_data["exact_synonyms"] == []
        assert synonym_data["narrow_synonyms"] == []
        assert synonym_data["broad_synonyms"] == []
        assert synonym_data["related_synonyms"] == []
        assert synonym_data["all_synonyms"] == []

    def test_malformed_synonym_handling(self):
        """Test handling of malformed synonym structures."""
        node_malformed_synonyms = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "synonyms": [
                    {"pred": "hasExactSynonym", "val": "good synonym"},
                    {"pred": "hasExactSynonym"},  # Missing val
                    {"val": "synonym without pred"},  # Missing pred
                    {},  # Empty synonym object
                    {"pred": "unknownSynonymType", "val": "unknown type"}  # Unknown type
                ]
            }
        }
        
        synonym_data = extract_synonyms_from_go_node(node_malformed_synonyms)
        
        # Should extract the good synonym
        assert "good synonym" in synonym_data["exact_synonyms"]
        assert "good synonym" in synonym_data["all_synonyms"]
        
        # Should handle malformed ones gracefully - parser is lenient and includes 
        # synonyms with text even if pred is missing or unknown
        assert len(synonym_data["all_synonyms"]) == 3  # good synonym + synonym without pred + unknown type
        assert "good synonym" in synonym_data["all_synonyms"]
        assert "synonym without pred" in synonym_data["all_synonyms"]
        assert "unknown type" in synonym_data["all_synonyms"]

    def test_synonym_case_preservation(self, diabetes_node):
        """Test that synonym case is preserved during extraction."""
        result = parse_enhanced_go_term(diabetes_node)
        
        # Should preserve exact case from source
        assert "NIDDM" in result["exact_synonyms"]  # All caps preserved
        assert "Type II diabetes mellitus" in result["exact_synonyms"]  # Title case preserved

    def test_synonym_whitespace_handling(self):
        """Test handling of synonyms with various whitespace patterns."""
        node_whitespace_synonyms = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "synonyms": [
                    {"pred": "hasExactSynonym", "val": "  leading spaces"},
                    {"pred": "hasExactSynonym", "val": "trailing spaces  "},
                    {"pred": "hasExactSynonym", "val": "  both sides  "},
                    {"pred": "hasExactSynonym", "val": "multiple  internal  spaces"},
                ]
            }
        }
        
        synonym_data = extract_synonyms_from_go_node(node_whitespace_synonyms)
        
        # Should preserve the synonyms as-is (parser doesn't trim)
        assert "  leading spaces" in synonym_data["exact_synonyms"]
        assert "trailing spaces  " in synonym_data["exact_synonyms"]
        assert "  both sides  " in synonym_data["exact_synonyms"]
        assert "multiple  internal  spaces" in synonym_data["exact_synonyms"]

    def test_ontology_manager_synonym_integration(self, ontology_manager, diabetes_node):
        """Test OntologyManager integration with synonym type parsing."""
        result = ontology_manager._extract_enhanced_term_data(diabetes_node)
        
        # Should have properly categorized synonyms
        assert len(result["exact_synonyms"]) >= 3
        assert len(result["narrow_synonyms"]) >= 1
        
        # Check specific expected synonyms
        assert "Type II diabetes mellitus" in result["exact_synonyms"]
        assert "maturity-onset diabetes of the young" in result["narrow_synonyms"]
        
        # Note: OntologyManager doesn't return related_synonyms separately,
        # but they should be in all_synonyms
        assert "adult-onset diabetes" in result["all_synonyms"]

    def test_synonym_type_statistics(self, sample_do_data):
        """Test statistics about synonym type distribution in test data."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        stats = {
            "exact": 0,
            "narrow": 0,
            "broad": 0,
            "related": 0,
            "total_nodes": 0,
            "nodes_with_synonyms": 0
        }
        
        for node in nodes:
            stats["total_nodes"] += 1
            synonym_data = extract_synonyms_from_go_node(node)
            
            if synonym_data["all_synonyms"]:
                stats["nodes_with_synonyms"] += 1
            
            stats["exact"] += len(synonym_data["exact_synonyms"])
            stats["narrow"] += len(synonym_data["narrow_synonyms"])
            stats["broad"] += len(synonym_data["broad_synonyms"])
            stats["related"] += len(synonym_data["related_synonyms"])
        
        # Basic sanity checks on test data
        assert stats["total_nodes"] > 0
        assert stats["nodes_with_synonyms"] > 0
        assert stats["exact"] > 0  # Should have exact synonyms
        assert stats["narrow"] > 0  # Should have narrow synonyms
        assert stats["related"] > 0  # Should have related synonyms
        
        # Print stats for debugging (will show in verbose test output)
        print(f"\nSynonym statistics: {stats}")

    def test_unicode_synonyms_handling(self):
        """Test handling of unicode characters in synonyms."""
        node_unicode_synonyms = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "synonyms": [
                    {"pred": "hasExactSynonym", "val": "maladie française"},
                    {"pred": "hasNarrowSynonym", "val": "enfermedad específica"},
                    {"pred": "hasBroadSynonym", "val": "болезнь"},
                    {"pred": "hasRelatedSynonym", "val": "疾病"}
                ]
            }
        }
        
        synonym_data = extract_synonyms_from_go_node(node_unicode_synonyms)
        
        # Should handle unicode correctly
        assert "maladie française" in synonym_data["exact_synonyms"]
        assert "enfermedad específica" in synonym_data["narrow_synonyms"]
        assert "болезнь" in synonym_data["broad_synonyms"]
        assert "疾病" in synonym_data["related_synonyms"]
        
        # All should be in all_synonyms
        assert len(synonym_data["all_synonyms"]) == 4