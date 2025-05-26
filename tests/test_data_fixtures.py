"""Test data fixtures for DO parser integration tests."""
import json
import pytest
from pathlib import Path
from typing import Dict, Any


class DOTestData:
    """Helper class for managing DO test data."""
    
    def __init__(self):
        self.test_data_dir = Path(__file__).parent / "data"
        
    def load_test_file(self, filename: str) -> Dict[str, Any]:
        """Load a test data file."""
        file_path = self.test_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @property
    def comprehensive_sample(self) -> Dict[str, Any]:
        """Load comprehensive sample with various DO term types."""
        return self.load_test_file("sample_do_comprehensive.json")
    
    @property
    def edge_cases_sample(self) -> Dict[str, Any]:
        """Load edge cases sample for testing robustness."""
        return self.load_test_file("sample_do_edge_cases.json")
    
    @property
    def malformed_sample(self) -> Dict[str, Any]:
        """Load malformed data for error handling tests."""
        return self.load_test_file("sample_do_malformed.json")
    
    @property
    def performance_sample(self) -> Dict[str, Any]:
        """Load large dataset for performance testing."""
        return self.load_test_file("sample_do_performance.json")
    
    @property
    def empty_sample(self) -> Dict[str, Any]:
        """Load empty dataset."""
        return self.load_test_file("sample_do_empty.json")
    
    @property
    def invalid_sample(self) -> Dict[str, Any]:
        """Load invalid JSON structure."""
        return self.load_test_file("sample_do_invalid.json")
    
    def get_sample_terms(self, sample_type: str = "comprehensive") -> list:
        """Get parsed terms from a specific sample."""
        if sample_type == "comprehensive":
            data = self.comprehensive_sample
        elif sample_type == "edge_cases":
            data = self.edge_cases_sample
        elif sample_type == "malformed":
            data = self.malformed_sample
        elif sample_type == "performance":
            data = self.performance_sample
        elif sample_type == "empty":
            data = self.empty_sample
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
            
        return data.get("graphs", [{}])[0].get("nodes", [])
    
    def get_term_by_id(self, term_id: str, sample_type: str = "comprehensive") -> Dict[str, Any]:
        """Get a specific term by ID from a sample."""
        terms = self.get_sample_terms(sample_type)
        for term in terms:
            if term.get("id", "").endswith(term_id):
                return term
        raise ValueError(f"Term {term_id} not found in {sample_type} sample")


# Global test data instance
test_data = DOTestData()


@pytest.fixture
def do_test_data():
    """Pytest fixture for DO test data."""
    return test_data


@pytest.fixture
def comprehensive_do_data():
    """Comprehensive DO sample data fixture."""
    return test_data.comprehensive_sample


@pytest.fixture
def edge_cases_do_data():
    """Edge cases DO sample data fixture."""
    return test_data.edge_cases_sample


@pytest.fixture
def malformed_do_data():
    """Malformed DO sample data fixture."""
    return test_data.malformed_sample


@pytest.fixture
def performance_do_data():
    """Performance test DO sample data fixture."""
    return test_data.performance_sample


@pytest.fixture
def sample_do_terms():
    """Get sample DO terms for testing."""
    return test_data.get_sample_terms("comprehensive")


@pytest.fixture
def angiosarcoma_term():
    """Get the angiosarcoma term for detailed testing."""
    return test_data.get_term_by_id("DOID_0001816", "comprehensive")


@pytest.fixture
def diabetes_term():
    """Get the type 2 diabetes term for detailed testing."""
    return test_data.get_term_by_id("DOID_9352", "comprehensive")


@pytest.fixture
def covid_term():
    """Get the COVID-19 term for detailed testing."""
    return test_data.get_term_by_id("DOID_0080600", "comprehensive")


@pytest.fixture
def minimal_term():
    """Get minimal term for edge case testing."""
    return test_data.get_term_by_id("DOID_0000000", "edge_cases")


@pytest.fixture
def unicode_term():
    """Get unicode term for encoding testing."""
    return test_data.get_term_by_id("DOID_0000005", "edge_cases")


# Test data validation helpers
def validate_do_json_structure(data: Dict[str, Any]) -> bool:
    """Validate that data follows DO JSON structure."""
    try:
        assert "graphs" in data
        assert isinstance(data["graphs"], list)
        assert len(data["graphs"]) > 0
        
        graph = data["graphs"][0]
        assert "nodes" in graph
        assert isinstance(graph["nodes"], list)
        
        return True
    except (AssertionError, KeyError, IndexError):
        return False


def validate_do_term_structure(term: Dict[str, Any]) -> bool:
    """Validate that a term follows DO term structure."""
    try:
        # Required fields
        assert "id" in term
        assert "lbl" in term
        assert "type" in term
        
        # Optional meta structure
        if "meta" in term:
            meta = term["meta"]
            assert isinstance(meta, dict)
            
            # Validate definition structure if present
            if "definition" in meta:
                definition = meta["definition"]
                if isinstance(definition, dict):
                    assert "val" in definition
                    
            # Validate synonyms structure if present
            if "synonyms" in meta:
                synonyms = meta["synonyms"]
                assert isinstance(synonyms, list)
                for syn in synonyms:
                    if isinstance(syn, dict):
                        assert "pred" in syn or "val" in syn
                        
        return True
    except (AssertionError, KeyError, TypeError):
        return False


# Expected synonym types for validation
VALID_SYNONYM_TYPES = {
    "hasExactSynonym",
    "hasNarrowSynonym", 
    "hasBroadSynonym",
    "hasRelatedSynonym"
}

# Expected cross-reference prefixes
VALID_XREF_PREFIXES = {
    "MESH", "ICD10CM", "ICD9CM", "NCI", "OMIM", 
    "SNOMEDCT_US_2023_03_01", "UMLS_CUI", "url"
}


def get_test_statistics() -> Dict[str, Any]:
    """Get statistics about test data coverage."""
    stats = {
        "total_files": 6,
        "comprehensive_terms": len(test_data.get_sample_terms("comprehensive")),
        "edge_case_terms": len(test_data.get_sample_terms("edge_cases")),
        "performance_terms": len(test_data.get_sample_terms("performance")),
        "synonym_types_covered": set(),
        "xref_prefixes_covered": set(),
        "unicode_coverage": False,
        "error_cases_covered": 0
    }
    
    # Analyze comprehensive sample
    for term in test_data.get_sample_terms("comprehensive"):
        meta = term.get("meta", {})
        
        # Count synonym types
        for syn in meta.get("synonyms", []):
            if isinstance(syn, dict) and "pred" in syn:
                stats["synonym_types_covered"].add(syn["pred"])
                
        # Count xref prefixes
        for xref in meta.get("xrefs", []):
            if isinstance(xref, dict) and "val" in xref:
                prefix = xref["val"].split(":")[0] if ":" in xref["val"] else ""
                if prefix:
                    stats["xref_prefixes_covered"].add(prefix)
    
    # Check unicode coverage
    unicode_term = test_data.get_term_by_id("DOID_0000005", "edge_cases")
    if any(ord(c) > 127 for c in unicode_term.get("lbl", "")):
        stats["unicode_coverage"] = True
    
    # Count error cases
    stats["error_cases_covered"] = len(test_data.get_sample_terms("malformed"))
    
    return stats


if __name__ == "__main__":
    # Print test data statistics
    stats = get_test_statistics()
    print("DO Test Data Coverage Statistics:")
    print(f"Total test files: {stats['total_files']}")
    print(f"Comprehensive sample terms: {stats['comprehensive_terms']}")
    print(f"Edge case terms: {stats['edge_case_terms']}")
    print(f"Performance test terms: {stats['performance_terms']}")
    print(f"Synonym types covered: {sorted(stats['synonym_types_covered'])}")
    print(f"Cross-reference prefixes covered: {sorted(stats['xref_prefixes_covered'])}")
    print(f"Unicode coverage: {stats['unicode_coverage']}")
    print(f"Error cases covered: {stats['error_cases_covered']}")
    
    # Validate all test files
    print("\nValidating test data files:")
    for file_type in ["comprehensive", "edge_cases", "performance"]:
        try:
            data = getattr(test_data, f"{file_type}_sample")
            valid = validate_do_json_structure(data)
            print(f"✅ {file_type}: {'Valid' if valid else 'Invalid'}")
        except Exception as e:
            print(f"❌ {file_type}: Error - {e}")