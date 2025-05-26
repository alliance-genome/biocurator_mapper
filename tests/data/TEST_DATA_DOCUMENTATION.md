# DO Parser Test Data Documentation

## Overview

This directory contains comprehensive test data for DO (Disease Ontology) parser integration testing. The test data covers various scenarios including normal cases, edge cases, error conditions, and performance testing.

## Test Data Files

### 1. `sample_do_comprehensive.json`
**Purpose**: Main test dataset with realistic DO terms covering all major features.

**Contents**: 6 DO terms including:
- **DOID:0001816** (angiosarcoma) - Rich term with all synonym types and multiple xrefs
- **DOID:14566** (disease of cellular proliferation) - Basic term with exact and broad synonyms
- **DOID:9352** (type 2 diabetes mellitus) - Complex term with many synonyms and extensive xrefs
- **DOID:0080600** (COVID-19) - Modern disease term with contemporary references
- **DOID:0001067** (open-angle glaucoma) - Medical condition with clinical terminology
- **DOID:0000001** (disease) - Root term with minimal structure

**Features Covered**:
- ✅ All synonym types (exact, narrow, broad, related)
- ✅ Multiple cross-reference types (MESH, ICD10CM, ICD9CM, NCI, OMIM, SNOMEDCT)
- ✅ Rich definitions with xrefs
- ✅ UMLS CUI references
- ✅ Disease ontology namespace
- ✅ Various term complexities

### 2. `sample_do_edge_cases.json`
**Purpose**: Edge cases and boundary conditions for robustness testing.

**Contents**: 8 edge case terms including:
- **DOID:0000000** - Minimal term (only ID, name, namespace)
- **DOID:0000002** - Empty definition
- **DOID:0000003** - Empty synonym values
- **DOID:0000004** - Obsolete/deprecated term
- **DOID:0000005** - Unicode characters (ñáéíóú αβγδε 中文 العربية ру́сский)
- **DOID:0000006** - Malformed cross-references
- **DOID:0000007** - Extremely long names/definitions
- **DOID:0000008** - All synonym types including unknown types

**Test Scenarios**:
- ✅ Missing optional fields
- ✅ Empty field values
- ✅ Unicode encoding handling
- ✅ Text length limits
- ✅ Obsolete term filtering
- ✅ Unknown synonym types
- ✅ Malformed references

### 3. `sample_do_malformed.json`
**Purpose**: Malformed data for error handling and parser robustness testing.

**Contents**: 6 malformed terms testing:
- **DOID:9999001** - Missing required 'lbl' field
- **Missing ID** - Term without 'id' field
- **DOID:9999003** - Invalid meta structure (string instead of object)
- **DOID:9999004** - Malformed definition (string instead of object)
- **DOID:9999005** - Invalid synonym structures
- **DOID:9999006** - Malformed xref structures (numbers, null values)

**Error Conditions**:
- ✅ Missing required fields
- ✅ Type mismatches
- ✅ Invalid nested structures
- ✅ Null/undefined values
- ✅ Mixed data types in arrays

### 4. `sample_do_performance.json`
**Purpose**: Large dataset for performance and scalability testing.

**Contents**: 100 generated DO terms with:
- Consistent structure across all terms
- Realistic definition lengths
- All four synonym types per term
- Multiple cross-references per term
- Performance IDs: DOID:8000001 - DOID:8000100

**Performance Metrics**:
- ✅ Memory usage with large datasets
- ✅ Parsing speed benchmarks
- ✅ Batch processing efficiency
- ✅ Embedding generation scalability

### 5. `sample_do_empty.json`
**Purpose**: Empty dataset edge case.

**Contents**: Valid JSON structure with empty graphs array.

**Tests**: 
- ✅ Empty input handling
- ✅ Graceful degradation
- ✅ No-crash behavior

### 6. `sample_do_invalid.json`
**Purpose**: Invalid JSON structure for format validation.

**Contents**: JSON with wrong top-level structure (missing "graphs" key).

**Tests**:
- ✅ Schema validation
- ✅ Format error handling
- ✅ Invalid structure detection

## Data Coverage Analysis

### Synonym Types Coverage
- ✅ `hasExactSynonym` - Exact matches
- ✅ `hasNarrowSynonym` - More specific terms  
- ✅ `hasBroadSynonym` - More general terms
- ✅ `hasRelatedSynonym` - Related but not exact
- ✅ `hasUnknownSynonym` - Invalid/unknown type for error testing

### Cross-Reference Prefixes Coverage
- ✅ **MESH** - Medical Subject Headings
- ✅ **ICD10CM** - International Classification of Diseases, 10th Revision
- ✅ **ICD9CM** - International Classification of Diseases, 9th Revision
- ✅ **NCI** - National Cancer Institute Thesaurus
- ✅ **OMIM** - Online Mendelian Inheritance in Man
- ✅ **SNOMEDCT_US** - SNOMED Clinical Terms
- ✅ **UMLS_CUI** - Unified Medical Language System Concept Unique Identifier
- ✅ **URL** - Web references

### Language/Encoding Coverage
- ✅ **ASCII** - Standard English characters
- ✅ **Latin Extended** - ñáéíóú
- ✅ **Greek** - αβγδε
- ✅ **Chinese** - 中文
- ✅ **Arabic** - العربية
- ✅ **Cyrillic** - ру́сский
- ✅ **Korean** - 특별한 질병

### Medical Domain Coverage
- ✅ **Oncology** - angiosarcoma, cellular proliferation
- ✅ **Endocrinology** - type 2 diabetes mellitus
- ✅ **Infectious Disease** - COVID-19
- ✅ **Ophthalmology** - open-angle glaucoma
- ✅ **General** - disease root term

## Usage in Tests

### Basic Usage
```python
from tests.test_data_fixtures import test_data

# Load comprehensive sample
comprehensive_data = test_data.comprehensive_sample

# Get specific terms
diabetes_term = test_data.get_term_by_id("DOID_9352")
covid_term = test_data.get_term_by_id("DOID_0080600")
```

### Pytest Fixtures
```python
def test_parsing(comprehensive_do_data):
    # Use comprehensive sample in tests
    parsed_terms = parse_go_json_enhanced(comprehensive_do_data)
    assert len(parsed_terms) == 6

def test_specific_term(diabetes_term):
    # Test specific term parsing
    assert diabetes_term["lbl"] == "type 2 diabetes mellitus"
```

### Error Testing
```python
def test_error_handling(malformed_do_data):
    # Test parser robustness with malformed data
    try:
        parse_go_json_enhanced(malformed_do_data)
    except Exception as e:
        # Verify appropriate error handling
        assert "required field" in str(e).lower()
```

## Data Validation

### Validation Functions
- `validate_do_json_structure()` - Validates overall JSON structure
- `validate_do_term_structure()` - Validates individual term structure
- `get_test_statistics()` - Provides coverage statistics

### Expected Patterns
- **Term IDs**: `http://purl.obolibrary.org/obo/DOID_XXXXXXX`
- **Namespaces**: `disease_ontology`
- **Types**: `CLASS`
- **Synonym Predicates**: `hasExactSynonym`, `hasNarrowSynonym`, etc.

## Test Data Generation

The performance test data is generated programmatically:

```python
# Generate 100 terms with consistent structure
for i in range(100):
    term_id = f'DOID_800{i:04d}'
    # ... create term structure
```

## Maintenance

### Adding New Test Cases
1. Add terms to appropriate sample file
2. Update `test_data_fixtures.py` if new patterns needed
3. Add corresponding pytest fixtures
4. Update this documentation

### Data Validation
Run `python tests/test_data_fixtures.py` to validate all test data files and see coverage statistics.

## Statistics Summary

```
Total test files: 6
Comprehensive sample terms: 6
Edge case terms: 8  
Performance test terms: 100
Total unique terms: 114

Synonym types covered: 5 (including unknown type)
Cross-reference prefixes covered: 8
Unicode encoding coverage: Yes
Error cases covered: 6
Medical domains covered: 5
```

This comprehensive test data ensures robust testing of the DO parser across all expected use cases, edge conditions, and error scenarios.