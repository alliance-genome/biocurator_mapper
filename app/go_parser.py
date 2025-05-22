"""
Enhanced GO.json parser that extracts all useful fields for semantic matching.
"""
from typing import Dict, List


def extract_synonyms_from_go_node(node: Dict) -> Dict[str, List[str]]:
    """Extract and categorize synonyms from GO node metadata."""
    meta = node.get("meta", {})
    synonyms = meta.get("synonyms", [])
    
    exact_synonyms = []
    narrow_synonyms = []
    broad_synonyms = []
    related_synonyms = []
    all_synonyms = []
    
    for syn in synonyms:
        syn_text = syn.get("val", "")
        syn_type = syn.get("pred", "")
        
        if syn_text:
            all_synonyms.append(syn_text)
            
            if syn_type == "hasExactSynonym":
                exact_synonyms.append(syn_text)
            elif syn_type == "hasNarrowSynonym":
                narrow_synonyms.append(syn_text)
            elif syn_type == "hasBroadSynonym":
                broad_synonyms.append(syn_text)
            elif syn_type == "hasRelatedSynonym":
                related_synonyms.append(syn_text)
    
    return {
        "exact_synonyms": exact_synonyms,
        "narrow_synonyms": narrow_synonyms,
        "broad_synonyms": broad_synonyms,
        "related_synonyms": related_synonyms,
        "all_synonyms": all_synonyms
    }


def extract_cross_references(node: Dict) -> List[str]:
    """Extract cross-references to other databases."""
    meta = node.get("meta", {})
    xrefs = []
    
    # Check definition xrefs
    definition = meta.get("definition", {})
    if isinstance(definition, dict) and "xrefs" in definition:
        xrefs.extend(definition["xrefs"])
    
    # Check for other xref sources in basicPropertyValues
    basic_props = meta.get("basicPropertyValues", [])
    for prop in basic_props:
        if "hasDbXref" in prop.get("pred", ""):
            xrefs.append(prop.get("val", ""))
    
    return xrefs


def get_ontology_namespace(node: Dict) -> str:
    """Extract ontology namespace (biological_process, molecular_function, cellular_component)."""
    meta = node.get("meta", {})
    basic_props = meta.get("basicPropertyValues", [])
    
    for prop in basic_props:
        if prop.get("pred") == "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace":
            return prop.get("val", "")
    
    return ""


def parse_enhanced_go_term(node: Dict, id_format: Dict = None) -> Dict:
    """Parse GO node into enhanced format with all useful fields."""
    if id_format is None:
        id_format = {"prefix_replacement": {"_": ":"}}
    
    try:
        id_uri = node["id"]
        name = node["lbl"]
    except KeyError:
        return None
    
    # Transform ID format
    term_id = id_uri.split("/")[-1]
    for old, new in id_format.get("prefix_replacement", {}).items():
        term_id = term_id.replace(old, new)
    
    # Extract definition
    definition = ""
    meta = node.get("meta", {})
    if "definition" in meta:
        def_obj = meta["definition"]
        if isinstance(def_obj, dict):
            definition = def_obj.get("val", "")
        else:
            definition = str(def_obj)
    
    # Extract synonyms
    synonym_data = extract_synonyms_from_go_node(node)
    
    # Extract cross-references
    xrefs = extract_cross_references(node)
    
    # Extract namespace
    namespace = get_ontology_namespace(node)
    
    # Build searchable text combining all textual content
    searchable_components = [
        name,
        definition,
    ]
    searchable_components.extend(synonym_data["all_synonyms"])
    searchable_text = " ".join(filter(None, searchable_components))
    
    return {
        "id": term_id,
        "name": name,
        "definition": definition,
        "exact_synonyms": synonym_data["exact_synonyms"],
        "narrow_synonyms": synonym_data["narrow_synonyms"],
        "broad_synonyms": synonym_data["broad_synonyms"],
        "related_synonyms": synonym_data["related_synonyms"],
        "all_synonyms": synonym_data["all_synonyms"],
        "cross_references": xrefs,
        "namespace": namespace,
        "searchable_text": searchable_text
    }


def parse_go_json_enhanced(go_data: Dict, id_format: Dict = None) -> List[Dict]:
    """Parse GO.json file extracting all useful fields for each term."""
    if id_format is None:
        id_format = {"prefix_replacement": {"_": ":"}}
    
    graphs = go_data.get("graphs", [])
    if not graphs:
        return []
    
    nodes = graphs[0].get("nodes", [])
    parsed_terms = []
    
    for node in nodes:
        if "lbl" in node and "id" in node:  # Only process terms with required fields
            enhanced_term = parse_enhanced_go_term(node, id_format)
            if enhanced_term:
                parsed_terms.append(enhanced_term)
    
    return parsed_terms


# Example usage and testing
if __name__ == "__main__":
    # Test with real GO data
    import json
    
    with open("go.json", "r") as f:
        go_data = json.load(f)
    
    print("Testing enhanced GO parsing...")
    enhanced_terms = parse_go_json_enhanced(go_data)
    
    print(f"Parsed {len(enhanced_terms)} enhanced terms")
    
    # Show example with synonyms
    for term in enhanced_terms[:5]:
        if term["all_synonyms"]:
            print(f"\n=== {term['id']}: {term['name']} ===")
            print(f"Definition: {term['definition'][:100]}...")
            print(f"Namespace: {term['namespace']}")
            print(f"Exact synonyms ({len(term['exact_synonyms'])}): {term['exact_synonyms']}")
            print(f"Narrow synonyms ({len(term['narrow_synonyms'])}): {term['narrow_synonyms']}")
            print(f"Cross-refs ({len(term['cross_references'])}): {term['cross_references'][:3]}")
            break