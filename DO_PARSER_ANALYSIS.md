# Disease Ontology (DO) Parser Implementation Analysis

## Overview

The DO parser implementation leverages a shared parser architecture that handles both Gene Ontology (GO) and Disease Ontology (DO) data, since both use the same OBO JSON format. This analysis documents the complete data flow, components, and integration points.

## Core Components

### 1. Primary Parser (`app/go_parser.py`)

**Main Functions:**
- `parse_go_json_enhanced(go_data, id_format)` - Entry point for parsing DO.json files
- `parse_enhanced_go_term(node, id_format)` - Processes individual DO terms
- `extract_synonyms_from_go_node(node)` - Categorizes synonyms by type
- `extract_cross_references(node)` - Extracts cross-references to other databases
- `get_ontology_namespace(node)` - Identifies ontology namespace (disease_ontology for DO)

**Key Features:**
- **ID Format Transformation**: Converts URI format (`http://purl.obolibrary.org/obo/DOID_0001816`) to standard format (`DOID:0001816`)
- **Synonym Categorization**: Extracts and categorizes synonyms into:
  - `hasExactSynonym` → `exact_synonyms`
  - `hasNarrowSynonym` → `narrow_synonyms` 
  - `hasBroadSynonym` → `broad_synonyms`
  - `hasRelatedSynonym` → `related_synonyms`
- **Cross-Reference Extraction**: Pulls xrefs from:
  - Definition xrefs (`meta.definition.xrefs`)
  - Basic property values with `hasDbXref` predicate
- **Namespace Detection**: Identifies DO terms by `disease_ontology` namespace
- **Searchable Text Building**: Combines name, definition, and all synonyms

### 2. Data Models (`app/models.py`)

**DOTerm Dataclass:**
```python
@dataclass
class DOTerm:
    id: str
    name: str
    definition: Optional[str] = None
    synonyms: Optional[Dict[str, List[str]]] = field(default_factory=dict)
```

**Note**: The current DOTerm model is simplified. The actual parsed data structure is richer and includes:
- Categorized synonyms (exact, narrow, broad, related)
- Cross-references
- Namespace information
- Searchable text

### 3. Ontology Manager Integration (`app/ontology_manager.py`)

**Key Method: `_extract_enhanced_term_data(raw_term)`**

This method serves as the bridge between parsed DO terms and Weaviate storage:

```python
def _extract_enhanced_term_data(self, raw_term: dict) -> dict:
    # If already parsed (has exact_synonyms), use directly
    if "exact_synonyms" in raw_term:
        return {
            "term_id": raw_term["id"],
            "name": raw_term["name"], 
            "definition": raw_term.get("definition", ""),
            "exact_synonyms": raw_term.get("exact_synonyms", []),
            "narrow_synonyms": raw_term.get("narrow_synonyms", []),
            "broad_synonyms": raw_term.get("broad_synonyms", []),
            "all_synonyms": raw_term.get("all_synonyms", []),
            "searchable_text": raw_term.get("searchable_text", "")
        }
    
    # If raw DO/GO node, parse using go_parser
    if "lbl" in raw_term and "meta" in raw_term:
        parsed_term = parse_enhanced_go_term(raw_term)
        # ... transform to Weaviate format
```

**Text Processing: `_build_searchable_text(term_data)`**

Builds comprehensive searchable text based on embeddings configuration:
- Includes name, definition, and synonyms based on `vectorize_fields` config
- Applies preprocessing (lowercase, punctuation removal)
- Uses configurable separator (default: " | ")

## Complete Data Flow

### 1. Download Phase (`app/main.py`)
```
URL: http://purl.obolibrary.org/obo/doid.json
↓
Storage: /data/ontologies/DOID/doid_YYYYMMDD_HHMMSS.json
↓
Symlink: /data/ontologies/DOID/doid_latest.json
```

### 2. Parsing Phase (`app/main.py` line 519)
```python
# Get DO-specific configuration
ontology_config = get_ontology_config("DOID")
id_format = ontology_config.get("id_format", {"prefix_replacement": {"_": ":"}})

# Parse using enhanced GO parser
parsed_terms = parse_go_json_enhanced(data, id_format)
```

**Configuration (`ontology_config.yaml`):**
```yaml
DOID:
  name: "Disease Ontology"
  description: "Human Disease Ontology"
  default_source_url: "http://purl.obolibrary.org/obo/doid.json"
  id_format:
    separator: ":"
    prefix_replacement:
      "_": ":"
  enabled: true
```

### 3. Embedding Phase (`app/ontology_manager.py`)
```
parsed_terms 
↓
_extract_enhanced_term_data() - Transform to Weaviate format
↓
_build_searchable_text() - Build comprehensive search text
↓
OpenAI Embedding Generation (via do_embeddings.py)
↓
Weaviate Collection Creation
```

**Weaviate Schema:**
- `term_id`: DO term identifier (e.g., "DOID:0001816")
- `name`: Primary term name
- `definition`: Term definition
- `exact_synonyms`: Array of exact synonyms
- `narrow_synonyms`: Array of narrow synonyms  
- `broad_synonyms`: Array of broad synonyms
- `searchable_text`: Combined text for vector search

### 4. Search Phase (`app/ontology_searcher.py`)
```
User Query
↓
OpenAI Query Embedding
↓
Weaviate Vector Search
↓
Enriched Results with Synonyms
```

## Configuration Integration

### DO-Specific Embeddings Config (`embeddings_config.yaml`)
```yaml
do_specific:
  synonym_types:
    exact_synonym: 1.0      # Highest weight
    narrow_synonym: 0.8     # High weight
    broad_synonym: 0.7      # Medium weight  
    related_synonym: 0.5    # Lower weight
  include_metadata:
    xref_sources: ["MESH", "ICD10CM", "SNOMEDCT", "OMIM"]
    definition_required: true
    include_obsolete: false
  text_composition:
    primary_text: "name"
    context_fields: ["definition", "synonyms"]
    separator: " | "
    max_text_length: 8000
  quality_filters:
    min_definition_length: 10
    exclude_patterns: ["deprecated", "obsolete"]
```

## Testing Architecture

### Existing Tests
1. **`test_doid_parsing.py`** - Tests DO compatibility with GO parser
2. **`test_ontology_manager_do_extraction.py`** - Tests DO term extraction
3. **`test_do_embeddings.py`** - Tests DO embedding generation

### Sample DO Data Structure
```json
{
  "graphs": [{
    "nodes": [{
      "id": "http://purl.obolibrary.org/obo/DOID_0001816",
      "lbl": "angiosarcoma", 
      "type": "CLASS",
      "meta": {
        "definition": {
          "val": "A malignant vascular tumor...",
          "xrefs": ["url:http://en.wikipedia.org/wiki/Hemangiosarcoma"]
        },
        "synonyms": [
          {"pred": "hasExactSynonym", "val": "hemangiosarcoma"},
          {"pred": "hasRelatedSynonym", "val": "malignant hemangioendothelioma"}
        ],
        "basicPropertyValues": [{
          "pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace",
          "val": "disease_ontology"
        }]
      }
    }]
  }]
}
```

## API Endpoints

### Admin Endpoints
- `POST /admin/update_ontology` - Download DO data
- `POST /admin/generate_embeddings` - Create DO embeddings  
- `GET /admin/download_status/{ontology_name}` - Check download status

### Search Endpoints
- `POST /resolve_biocurated_data` - Main resolution endpoint using DO data

## Error Handling & Edge Cases

### Current Handling
- **Missing Fields**: Graceful fallback to empty values
- **Malformed JSON**: Exception propagation with logging
- **Missing Synonyms**: Empty arrays returned
- **Invalid IDs**: Basic validation in ID transformation

### Potential Issues
- **Large Files**: Memory usage with full JSON loading
- **Encoding Issues**: UTF-8 handling in definitions
- **Circular References**: Not explicitly handled
- **Version Compatibility**: OBO format changes over time

## Performance Characteristics

### Memory Usage
- Full DO.json loaded into memory (~50-100MB)
- Parsed terms stored as Python objects
- Weaviate batch processing reduces memory pressure

### Processing Speed
- Parsing: ~1-2 seconds for full DO
- Embedding: Dependent on OpenAI API rate limits
- Search: Sub-second with Weaviate vector search

## Integration Points Summary

1. **File System**: `/data/ontologies/DOID/` for JSON storage
2. **Configuration**: `ontology_config.yaml` and `embeddings_config.yaml`
3. **Parsing**: `go_parser.py` (shared with GO)
4. **Processing**: `ontology_manager.py` (_extract_enhanced_term_data)
5. **Embedding**: `do_embeddings.py` (DO-specific workflow)
6. **Storage**: Weaviate collections with vector embeddings
7. **Search**: `ontology_searcher.py` for semantic search
8. **API**: FastAPI endpoints for download, embedding, and search

This architecture provides a robust, configurable system for processing Disease Ontology data from raw JSON through to semantic search capabilities.