"""Test DO term extraction in OntologyManager."""
import pytest

from app.ontology_manager import OntologyManager


class TestOntologyManagerDOExtraction:
    """Test that OntologyManager correctly extracts DO term data."""

    @pytest.fixture
    def ontology_manager(self):
        """Create an OntologyManager instance."""
        return OntologyManager()

    @pytest.fixture
    def sample_do_node(self):
        """Sample DO node in raw JSON format."""
        return {
            "id": "http://purl.obolibrary.org/obo/DOID_0001816",
            "lbl": "angiosarcoma",
            "type": "CLASS",
            "meta": {
                "definition": {
                    "val": "A malignant vascular tumor that results_in rapidly proliferating, extensively infiltrating anaplastic cells derived_from blood vessels and derived_from the lining of irregular blood-filled spaces.",
                    "xrefs": ["url:http://en.wikipedia.org/wiki/Hemangiosarcoma"]
                },
                "synonyms": [
                    {
                        "pred": "hasExactSynonym",
                        "val": "hemangiosarcoma"
                    },
                    {
                        "pred": "hasNarrowSynonym",
                        "val": "blood vessel cancer"
                    },
                    {
                        "pred": "hasBroadSynonym",
                        "val": "vascular tumor"
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
        }

    @pytest.fixture
    def sample_parsed_do_term(self):
        """Sample already parsed DO term."""
        return {
            "id": "DOID:0050686",
            "name": "organ system cancer",
            "definition": "A cancer that is classified based on the organ it starts in.",
            "exact_synonyms": ["organ cancer"],
            "narrow_synonyms": ["specific organ cancer"],
            "broad_synonyms": ["cancer"],
            "all_synonyms": ["organ cancer", "specific organ cancer", "cancer"],
            "searchable_text": "organ system cancer A cancer that is classified based on the organ it starts in. organ cancer specific organ cancer cancer"
        }

    def test_extract_raw_do_node(self, ontology_manager, sample_do_node):
        """Test extraction from raw DO node format."""
        result = ontology_manager._extract_enhanced_term_data(sample_do_node)

        # Check basic fields
        assert result["term_id"] == "DOID:0001816"
        assert result["name"] == "angiosarcoma"
        assert "malignant vascular tumor" in result["definition"]

        # Check synonym extraction
        assert "hemangiosarcoma" in result["exact_synonyms"]
        assert len(result["exact_synonyms"]) == 1

        assert "blood vessel cancer" in result["narrow_synonyms"]
        assert len(result["narrow_synonyms"]) == 1

        assert "vascular tumor" in result["broad_synonyms"]
        assert len(result["broad_synonyms"]) == 1

        # Check all_synonyms contains all types
        assert len(result["all_synonyms"]) == 4
        assert "hemangiosarcoma" in result["all_synonyms"]
        assert "blood vessel cancer" in result["all_synonyms"]
        assert "vascular tumor" in result["all_synonyms"]
        assert "malignant hemangioendothelioma" in result["all_synonyms"]

    def test_extract_parsed_do_term(self, ontology_manager, sample_parsed_do_term):
        """Test extraction from already parsed DO term."""
        result = ontology_manager._extract_enhanced_term_data(sample_parsed_do_term)

        # Should preserve all fields
        assert result["term_id"] == "DOID:0050686"
        assert result["name"] == "organ system cancer"
        assert result["definition"] == "A cancer that is classified based on the organ it starts in."
        assert result["exact_synonyms"] == ["organ cancer"]
        assert result["narrow_synonyms"] == ["specific organ cancer"]
        assert result["broad_synonyms"] == ["cancer"]
        assert result["all_synonyms"] == ["organ cancer", "specific organ cancer", "cancer"]
        assert result["searchable_text"] == sample_parsed_do_term["searchable_text"]

    def test_extract_minimal_do_node(self, ontology_manager):
        """Test extraction from DO node with minimal data."""
        minimal_node = {
            "id": "http://purl.obolibrary.org/obo/DOID_12345",
            "lbl": "test disease",
            "meta": {
                # No definition
                "synonyms": []  # No synonyms
            }
        }

        result = ontology_manager._extract_enhanced_term_data(minimal_node)

        assert result["term_id"] == "DOID:12345"
        assert result["name"] == "test disease"
        assert result["definition"] == ""
        assert result["exact_synonyms"] == []
        assert result["narrow_synonyms"] == []
        assert result["broad_synonyms"] == []
        assert result["all_synonyms"] == []

    def test_extract_missing_fields_fallback(self, ontology_manager):
        """Test fallback behavior for non-standard format."""
        non_standard_term = {
            "id": "DOID:99999",
            "name": "fallback disease",
            "definition": "A disease for testing fallback"
        }

        result = ontology_manager._extract_enhanced_term_data(non_standard_term)

        # Should use fallback extraction
        assert result["term_id"] == "DOID:99999"
        assert result["name"] == "fallback disease"
        assert result["definition"] == "A disease for testing fallback"
        # Synonyms should be empty in fallback
        assert result["exact_synonyms"] == []
        assert result["narrow_synonyms"] == []
        assert result["broad_synonyms"] == []
        assert result["all_synonyms"] == []

    def test_build_searchable_text_with_do_data(self, ontology_manager):
        """Test building searchable text for DO terms."""
        term_data = {
            "name": "lung cancer",
            "definition": "A respiratory system cancer that is located_in the lung.",
            "exact_synonyms": ["cancer of lung", "lung neoplasm"],
            "narrow_synonyms": ["small cell lung cancer", "non-small cell lung cancer"],
            "broad_synonyms": ["respiratory cancer"],
            "all_synonyms": ["cancer of lung", "lung neoplasm", "small cell lung cancer",
                            "non-small cell lung cancer", "respiratory cancer"]
        }

        searchable_text = ontology_manager._build_searchable_text(term_data)

        # Should contain name and definition
        assert "lung cancer" in searchable_text
        assert "respiratory system cancer" in searchable_text

        # Should contain all synonym types
        assert "cancer of lung" in searchable_text
        assert "small cell lung cancer" in searchable_text
        assert "respiratory cancer" in searchable_text

