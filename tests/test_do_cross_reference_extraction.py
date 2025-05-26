"""Test DO cross-reference extraction from various sources."""
import json
import pytest
from pathlib import Path

from app.go_parser import extract_cross_references, parse_enhanced_go_term
from app.ontology_manager import OntologyManager


class TestDOCrossReferenceExtraction:
    """Test comprehensive cross-reference extraction for DO terms."""

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
    def diabetes_node(self, sample_do_data):
        """Get the type 2 diabetes node which has many cross-references."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "type 2 diabetes" in node["lbl"])

    @pytest.fixture
    def angiosarcoma_node(self, sample_do_data):
        """Get the angiosarcoma node with specific cross-references."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "angiosarcoma" in node["lbl"])

    @pytest.fixture
    def covid_node(self, sample_do_data):
        """Get the COVID-19 node with cross-references."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        return next(node for node in nodes if "COVID-19" in node["lbl"])

    def test_extract_main_xrefs_section(self, diabetes_node):
        """Test extraction from main xrefs section in meta."""
        xrefs = extract_cross_references(diabetes_node)
        
        # Should extract cross-references from main xrefs section
        assert len(xrefs) > 0
        
        # Check for specific expected cross-references
        expected_xrefs = [
            "ICD10CM:E11",
            "ICD9CM:250.00", 
            "MESH:D003924",
            "NCI:C26747",
            "OMIM:125853",
            "SNOMEDCT_US_2023_03_01:44054006"
        ]
        
        for expected in expected_xrefs:
            assert expected in xrefs, f"Missing expected xref: {expected}"

    def test_extract_basicpropertyvalues_xrefs(self, diabetes_node):
        """Test extraction from basicPropertyValues section."""
        xrefs = extract_cross_references(diabetes_node)
        
        # Should extract UMLS cross-reference from basicPropertyValues
        umls_refs = [xref for xref in xrefs if "UMLS_CUI:" in xref]
        assert len(umls_refs) > 0
        assert "UMLS_CUI:C0011860" in umls_refs

    def test_extract_definition_xrefs(self, angiosarcoma_node):
        """Test extraction from definition xrefs."""
        xrefs = extract_cross_references(angiosarcoma_node)
        
        # Should extract definition xrefs (URLs in this case)
        url_refs = [xref for xref in xrefs if "url:" in xref or "http" in xref]
        assert len(url_refs) > 0

    def test_database_specific_xref_extraction(self, diabetes_node):
        """Test extraction of specific database cross-references."""
        xrefs = extract_cross_references(diabetes_node)
        
        # Test different database types
        icd10_refs = [xref for xref in xrefs if "ICD10CM:" in xref]
        icd9_refs = [xref for xref in xrefs if "ICD9CM:" in xref]
        mesh_refs = [xref for xref in xrefs if "MESH:" in xref]
        nci_refs = [xref for xref in xrefs if "NCI:" in xref]
        omim_refs = [xref for xref in xrefs if "OMIM:" in xref]
        snomed_refs = [xref for xref in xrefs if "SNOMEDCT" in xref]
        
        assert len(icd10_refs) > 0, "Should have ICD10 references"
        assert len(icd9_refs) > 0, "Should have ICD9 references"
        assert len(mesh_refs) > 0, "Should have MESH references"
        assert len(nci_refs) > 0, "Should have NCI references"
        assert len(omim_refs) > 0, "Should have OMIM references"
        assert len(snomed_refs) > 0, "Should have SNOMED references"

    def test_xref_format_validation(self, diabetes_node):
        """Test that cross-references follow expected formats."""
        xrefs = extract_cross_references(diabetes_node)
        
        for xref in xrefs:
            # Each xref should be a non-empty string
            assert isinstance(xref, str)
            assert len(xref) > 0
            
            # Should not contain obvious parsing errors
            assert xref != "None"
            assert xref != ""

    def test_parse_enhanced_go_term_xref_integration(self, diabetes_node):
        """Test cross-reference integration in full term parsing."""
        result = parse_enhanced_go_term(diabetes_node)
        
        # Should have cross_references field
        assert "cross_references" in result
        assert isinstance(result["cross_references"], list)
        assert len(result["cross_references"]) > 0
        
        # Should contain expected types
        xrefs = result["cross_references"]
        has_mesh = any("MESH:" in xref for xref in xrefs)
        has_icd = any("ICD" in xref for xref in xrefs)
        has_snomed = any("SNOMEDCT" in xref for xref in xrefs)
        
        assert has_mesh, "Should have MESH references"
        assert has_icd, "Should have ICD references"
        assert has_snomed, "Should have SNOMED references"

    def test_empty_xrefs_handling(self):
        """Test handling of nodes with no cross-references."""
        node_no_xrefs = {
            "id": "http://purl.obolibrary.org/obo/DOID_0000001",
            "lbl": "disease",
            "meta": {
                "definition": {
                    "val": "A disposition to undergo pathological processes."
                }
                # No xrefs, no basicPropertyValues with xrefs
            }
        }
        
        xrefs = extract_cross_references(node_no_xrefs)
        assert xrefs == []

    def test_malformed_xrefs_handling(self):
        """Test handling of malformed cross-reference structures."""
        node_malformed_xrefs = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "xrefs": [
                    {"val": "MESH:D123456"},  # Good xref
                    {},  # Empty xref object
                    {"val": ""},  # Empty val
                    {"notval": "ICD10:X123"},  # Wrong key
                    "DIRECT_STRING_XREF",  # String instead of object
                ],
                "basicPropertyValues": [
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                        "val": "UMLS_CUI:C123456"
                    },
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
                        # Missing val
                    }
                ]
            }
        }
        
        xrefs = extract_cross_references(node_malformed_xrefs)
        
        # Should extract the good ones
        assert "MESH:D123456" in xrefs
        assert "DIRECT_STRING_XREF" in xrefs
        assert "UMLS_CUI:C123456" in xrefs
        
        # Should have filtered out malformed ones
        assert "" not in xrefs
        assert "ICD10:X123" not in xrefs  # Wrong key format

    def test_comprehensive_xref_extraction_all_nodes(self, sample_do_data):
        """Test cross-reference extraction across all nodes in test data."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        total_xrefs = 0
        nodes_with_xrefs = 0
        xref_databases = set()
        
        for node in nodes:
            xrefs = extract_cross_references(node)
            
            if xrefs:
                nodes_with_xrefs += 1
                total_xrefs += len(xrefs)
                
                # Collect database prefixes
                for xref in xrefs:
                    if ":" in xref and not xref.startswith("http"):
                        prefix = xref.split(":")[0]
                        xref_databases.add(prefix)
        
        # Basic sanity checks
        assert nodes_with_xrefs > 0, "Should have nodes with cross-references"
        assert total_xrefs > 0, "Should have extracted cross-references"
        assert len(xref_databases) > 3, "Should have multiple database types"
        
        # Print stats for debugging
        print(f"\nCross-reference statistics:")
        print(f"Nodes with xrefs: {nodes_with_xrefs}/{len(nodes)}")
        print(f"Total xrefs: {total_xrefs}")
        print(f"Database types: {sorted(xref_databases)}")

    def test_xref_deduplication(self):
        """Test that duplicate cross-references are handled properly."""
        node_duplicate_xrefs = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "xrefs": [
                    {"val": "MESH:D123456"},
                    {"val": "MESH:D123456"},  # Duplicate
                    {"val": "ICD10:E123"}
                ],
                "basicPropertyValues": [
                    {
                        "pred": "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
                        "val": "MESH:D123456"  # Duplicate from different source
                    }
                ]
            }
        }
        
        xrefs = extract_cross_references(node_duplicate_xrefs)
        
        # Current implementation doesn't deduplicate - this is expected behavior
        # as different sources might have the same xref
        mesh_count = xrefs.count("MESH:D123456")
        assert mesh_count == 3  # From all three sources

    def test_url_xref_extraction(self, angiosarcoma_node):
        """Test extraction of URL-based cross-references."""
        xrefs = extract_cross_references(angiosarcoma_node)
        
        # Should extract URL references from definition
        url_refs = [xref for xref in xrefs if "url:" in xref or "http" in xref]
        assert len(url_refs) > 0
        
        # Check for specific URL patterns
        has_wikipedia = any("wikipedia" in xref.lower() for xref in url_refs)
        has_ncit = any("ncit.nci.nih.gov" in xref for xref in url_refs)
        
        # At least one of these should be present based on test data
        assert has_wikipedia or has_ncit

    def test_special_character_xrefs(self):
        """Test handling of cross-references with special characters."""
        node_special_xrefs = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                "xrefs": [
                    {"val": "SNOMEDCT_US_2023_03_01:44054006"},  # Underscores and dates
                    {"val": "ICD10CM:E11.9"},  # Periods
                    {"val": "UMLS_CUI:C0011860"},  # Underscores
                    {"val": "MSH:D003924"},  # Different prefix format
                ]
            }
        }
        
        xrefs = extract_cross_references(node_special_xrefs)
        
        # Should handle special characters properly
        assert "SNOMEDCT_US_2023_03_01:44054006" in xrefs
        assert "ICD10CM:E11.9" in xrefs
        assert "UMLS_CUI:C0011860" in xrefs
        assert "MSH:D003924" in xrefs

    def test_ontology_manager_xref_integration(self, ontology_manager, diabetes_node):
        """Test that OntologyManager preserves cross-references."""
        # Note: Current OntologyManager doesn't expose cross_references directly
        # but the go_parser integration should work
        parsed_result = parse_enhanced_go_term(diabetes_node)
        
        # Verify parser extracted cross-references
        assert "cross_references" in parsed_result
        assert len(parsed_result["cross_references"]) > 0
        
        # Test OntologyManager processing
        result = ontology_manager._extract_enhanced_term_data(diabetes_node)
        
        # OntologyManager focuses on term data, not cross-references
        # but the underlying parsing should work
        assert "term_id" in result
        assert result["term_id"] == "DOID:9352"

    def test_performance_xref_extraction(self, sample_do_data):
        """Test performance of cross-reference extraction."""
        import time
        
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        start_time = time.time()
        
        for node in nodes:
            xrefs = extract_cross_references(node)
            # Process each node
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process reasonably quickly
        assert processing_time < 2.0, f"Xref extraction took too long: {processing_time:.2f} seconds"

    def test_xref_count_validation(self, sample_do_data):
        """Test that cross-reference counts are reasonable."""
        nodes = sample_do_data["graphs"][0]["nodes"]
        
        for node in nodes:
            xrefs = extract_cross_references(node)
            
            # Reasonable limits - shouldn't have too many or negative counts
            assert len(xrefs) >= 0
            assert len(xrefs) < 100  # Sanity check - shouldn't have hundreds of xrefs
            
            # If there are xrefs, they should be meaningful
            if xrefs:
                for xref in xrefs:
                    assert len(xref) > 2  # At least "X:Y" format or longer