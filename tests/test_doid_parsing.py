"""Test DOID (Disease Ontology) parsing compatibility with GO parser."""
import pytest
from app.go_parser import parse_go_json_enhanced, parse_enhanced_go_term


class TestDOIDParsing:
    """Test that DOID can be parsed using the GO parser."""
    
    @pytest.fixture
    def sample_doid_data(self):
        """Sample DOID data in OBO JSON format."""
        return {
            "graphs": [{
                "nodes": [
                    {
                        "id": "http://purl.obolibrary.org/obo/DOID_0001816",
                        "lbl": "angiosarcoma",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "A malignant vascular tumor that results_in rapidly proliferating, extensively infiltrating anaplastic cells derived_from blood vessels and derived_from the lining of irregular blood-filled spaces.",
                                "xrefs": ["url:http://en.wikipedia.org/wiki/Hemangiosarcoma", "url:http://ncit.nci.nih.gov/ncitbrowser/ConceptReport.jsp?dictionary=NCI_Thesaurus&version=14.10d&code=C3088"]
                            },
                            "synonyms": [
                                {
                                    "pred": "hasExactSynonym",
                                    "val": "hemangiosarcoma"
                                },
                                {
                                    "pred": "hasRelatedSynonym",
                                    "val": "malignant hemangioendothelioma"
                                }
                            ],
                            "basicPropertyValues": [
                                {
                                    "pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
                                    "val": "disease_ontology"
                                }
                            ]
                        }
                    },
                    {
                        "id": "http://purl.obolibrary.org/obo/DOID_14566",
                        "lbl": "disease of cellular proliferation",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "A disease that is characterized by abnormally rapid cell division.",
                                "xrefs": ["url:http://en.wikipedia.org/wiki/Cell_proliferation"]
                            },
                            "synonyms": [
                                {
                                    "pred": "hasExactSynonym",
                                    "val": "cell proliferation disease"
                                }
                            ]
                        }
                    }
                ]
            }]
        }
    
    def test_parse_single_doid_term(self, sample_doid_data):
        """Test parsing a single DOID term using GO parser."""
        node = sample_doid_data["graphs"][0]["nodes"][0]
        
        # Parse using the enhanced GO parser
        parsed_term = parse_enhanced_go_term(node)
        
        assert parsed_term is not None
        assert parsed_term["id"] == "DOID:0001816"
        assert parsed_term["name"] == "angiosarcoma"
        assert "malignant vascular tumor" in parsed_term["definition"]
        
        # Check synonyms
        assert len(parsed_term["exact_synonyms"]) == 1
        assert "hemangiosarcoma" in parsed_term["exact_synonyms"]
        assert len(parsed_term["related_synonyms"]) == 1
        assert "malignant hemangioendothelioma" in parsed_term["related_synonyms"]
        
        # Check searchable text includes all components
        assert "angiosarcoma" in parsed_term["searchable_text"]
        assert "hemangiosarcoma" in parsed_term["searchable_text"]
        assert "malignant vascular tumor" in parsed_term["searchable_text"]
    
    def test_parse_multiple_doid_terms(self, sample_doid_data):
        """Test parsing multiple DOID terms."""
        parsed_terms = parse_go_json_enhanced(sample_doid_data)
        
        assert len(parsed_terms) == 2
        
        # Check first term
        term1 = next(t for t in parsed_terms if t["id"] == "DOID:0001816")
        assert term1["name"] == "angiosarcoma"
        assert "hemangiosarcoma" in term1["all_synonyms"]
        
        # Check second term
        term2 = next(t for t in parsed_terms if t["id"] == "DOID:14566")
        assert term2["name"] == "disease of cellular proliferation"
        assert "cell proliferation disease" in term2["all_synonyms"]
    
    def test_doid_id_format_conversion(self, sample_doid_data):
        """Test conversion of DOID URIs to standard DOID:XXXXXXX format."""
        nodes = sample_doid_data["graphs"][0]["nodes"]
        
        expected_conversions = [
            ("http://purl.obolibrary.org/obo/DOID_0001816", "DOID:0001816"),
            ("http://purl.obolibrary.org/obo/DOID_14566", "DOID:14566")
        ]
        
        for i, (uri, expected_id) in enumerate(expected_conversions):
            node = nodes[i]
            parsed = parse_enhanced_go_term(node)
            assert parsed["id"] == expected_id
    
    def test_doid_namespace_extraction(self, sample_doid_data):
        """Test namespace extraction for DOID terms."""
        node = sample_doid_data["graphs"][0]["nodes"][0]
        parsed = parse_enhanced_go_term(node)
        
        # DOID uses "disease_ontology" as namespace
        assert parsed["namespace"] == "disease_ontology"
    
    def test_doid_searchable_text_building(self, sample_doid_data):
        """Test that searchable text is properly built for DOID terms."""
        node = sample_doid_data["graphs"][0]["nodes"][0]
        parsed = parse_enhanced_go_term(node)
        
        searchable = parsed["searchable_text"]
        
        # Should contain name
        assert "angiosarcoma" in searchable
        
        # Should contain definition
        assert "malignant vascular tumor" in searchable
        
        # Should contain synonyms
        assert "hemangiosarcoma" in searchable
        assert "malignant hemangioendothelioma" in searchable
        
        # Should be space-separated
        assert len(searchable.split()) > 10  # Has multiple words
    
    def test_doid_cross_references(self, sample_doid_data):
        """Test extraction of cross-references from DOID terms."""
        node = sample_doid_data["graphs"][0]["nodes"][0]
        parsed = parse_enhanced_go_term(node)
        
        # Should have extracted xrefs from definition
        assert len(parsed["cross_references"]) == 2
        assert any("wikipedia" in xref for xref in parsed["cross_references"])
        assert any("ncit.nci.nih.gov" in xref for xref in parsed["cross_references"])