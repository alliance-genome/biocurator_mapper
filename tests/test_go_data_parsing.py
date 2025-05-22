import json
import pytest


def get_nested_value(data: dict, path: list, default=""):
    """Get nested value from dict using path list."""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def parse_go_term(node: dict, id_format: dict = None):
    """Parse a GO term node into our standard format."""
    if id_format is None:
        id_format = {"prefix_replacement": {"_": ":"}}
    
    try:
        id_uri = node["id"]
        name = node["lbl"]
    except KeyError:
        return None
    
    # Apply ID format transformation
    term_id = id_uri.split("/")[-1]
    for old, new in id_format.get("prefix_replacement", {}).items():
        term_id = term_id.replace(old, new)
    
    definition = get_nested_value(node, ["meta", "definition", "val"])
    
    return {
        "id": term_id,
        "name": name,
        "definition": definition
    }


class TestGODataParsing:
    """Test GO data parsing with real ontology data."""

    @pytest.fixture
    def real_go_data(self):
        """Real GO data extracted from actual GO.json file."""
        return {
            "graphs": [{
                "nodes": [
                    {
                        "id": "http://purl.obolibrary.org/obo/GO_0000001",
                        "lbl": "mitochondrion inheritance",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton.",
                                "xrefs": ["GOC:mcc", "PMID:10873824", "PMID:11389764"]
                            }
                        }
                    },
                    {
                        "id": "http://purl.obolibrary.org/obo/GO_0000002",
                        "lbl": "mitochondrial genome maintenance",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "The maintenance of the structure and integrity of the mitochondrial genome; includes replication and DNA repair."
                            }
                        }
                    },
                    {
                        "id": "http://purl.obolibrary.org/obo/GO_0000006",
                        "lbl": "high-affinity zinc transmembrane transporter activity",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "Enables the transfer of zinc ions (Zn2+) from one side of a membrane to the other, probably powered by ATP hydrolysis."
                            }
                        }
                    },
                    {
                        "id": "http://purl.obolibrary.org/obo/GO_0000003",
                        "lbl": "obsolete reproduction",
                        "type": "CLASS",
                        "meta": {
                            "definition": {
                                "val": "OBSOLETE. The production of new individuals that contain some portion of genetic material inherited from their parents."
                            }
                        }
                    }
                ]
            }]
        }

    def test_nested_value_extraction_with_real_data(self, real_go_data):
        """Test nested value extraction with real GO structure."""
        node = real_go_data["graphs"][0]["nodes"][0]
        
        # Test successful extraction
        definition = get_nested_value(node, ["meta", "definition", "val"])
        assert "mitochondria" in definition
        assert "daughter cells" in definition
        
        # Test missing path
        missing = get_nested_value(node, ["meta", "missing", "path"], "default")
        assert missing == "default"
        
        # Test partial path exists
        meta = get_nested_value(node, ["meta"])
        assert isinstance(meta, dict)
        assert "definition" in meta

    def test_go_id_format_conversion(self, real_go_data):
        """Test conversion of GO URIs to standard GO:XXXXXXX format."""
        nodes = real_go_data["graphs"][0]["nodes"]
        
        expected_conversions = [
            ("http://purl.obolibrary.org/obo/GO_0000001", "GO:0000001"),
            ("http://purl.obolibrary.org/obo/GO_0000002", "GO:0000002"),
            ("http://purl.obolibrary.org/obo/GO_0000006", "GO:0000006"),
            ("http://purl.obolibrary.org/obo/GO_0000003", "GO:0000003")
        ]
        
        for i, (uri, expected_id) in enumerate(expected_conversions):
            node = nodes[i]
            assert node["id"] == uri
            
            # Test our conversion logic
            term_id = node["id"].split("/")[-1].replace("_", ":")
            assert term_id == expected_id

    def test_parse_single_go_term(self, real_go_data):
        """Test parsing a single GO term."""
        node = real_go_data["graphs"][0]["nodes"][0]  # mitochondrion inheritance
        
        parsed_term = parse_go_term(node)
        
        assert parsed_term is not None
        assert parsed_term["id"] == "GO:0000001"
        assert parsed_term["name"] == "mitochondrion inheritance"
        assert "mitochondria" in parsed_term["definition"]
        assert "daughter cells" in parsed_term["definition"]
        assert len(parsed_term["definition"]) > 100

    def test_parse_multiple_go_terms(self, real_go_data):
        """Test parsing multiple GO terms."""
        nodes = real_go_data["graphs"][0]["nodes"]
        parsed_terms = []
        
        for node in nodes:
            parsed_term = parse_go_term(node)
            if parsed_term:
                parsed_terms.append(parsed_term)
        
        assert len(parsed_terms) == 4
        
        # Check specific terms
        term_dict = {term["id"]: term for term in parsed_terms}
        
        # Test mitochondrion inheritance
        mito_term = term_dict["GO:0000001"]
        assert mito_term["name"] == "mitochondrion inheritance"
        assert "mitochondria" in mito_term["definition"]
        
        # Test mitochondrial genome maintenance
        genome_term = term_dict["GO:0000002"]
        assert genome_term["name"] == "mitochondrial genome maintenance"
        assert "genome" in genome_term["definition"]
        
        # Test zinc transporter
        zinc_term = term_dict["GO:0000006"]
        assert "zinc" in zinc_term["name"]
        assert "transporter" in zinc_term["name"]

    def test_biological_content_validation(self, real_go_data):
        """Test that parsed terms contain expected biological content."""
        nodes = real_go_data["graphs"][0]["nodes"]
        
        # Test mitochondrion inheritance term (GO:0000001)
        mito_node = nodes[0]
        parsed = parse_go_term(mito_node)
        
        assert "mitochondrion" in parsed["name"]
        definition = parsed["definition"].lower()
        assert "mitochondria" in definition
        assert "daughter cells" in definition
        assert "mitosis" in definition or "meiosis" in definition
        assert "cytoskeleton" in definition
        
        # Test zinc transporter term (GO:0000006)
        zinc_node = nodes[2]
        parsed = parse_go_term(zinc_node)
        
        assert "zinc" in parsed["name"]
        assert "transporter" in parsed["name"]
        definition = parsed["definition"].lower()
        assert "zinc ions" in definition
        assert "membrane" in definition
        assert "atp" in definition

    def test_obsolete_term_handling(self, real_go_data):
        """Test handling of obsolete GO terms."""
        # GO:0000003 is obsolete reproduction
        obsolete_node = real_go_data["graphs"][0]["nodes"][3]
        parsed = parse_go_term(obsolete_node)
        
        assert parsed["id"] == "GO:0000003"
        assert "obsolete" in parsed["name"].lower()
        assert "OBSOLETE" in parsed["definition"]

    def test_term_structure_consistency(self, real_go_data):
        """Test that all parsed terms have consistent structure."""
        nodes = real_go_data["graphs"][0]["nodes"]
        
        for node in nodes:
            parsed = parse_go_term(node)
            assert parsed is not None
            
            # Check required fields
            assert "id" in parsed
            assert "name" in parsed
            assert "definition" in parsed
            
            # Check format
            assert parsed["id"].startswith("GO:")
            assert len(parsed["id"]) == 10  # GO:0000XXX format
            assert len(parsed["name"]) > 0
            assert len(parsed["definition"]) > 0
            
            # Check ID format
            assert parsed["id"].count(":") == 1
            assert parsed["id"].split(":")[1].isdigit()

    def test_json_structure_validation(self, real_go_data):
        """Test that the JSON structure matches expected GO format."""
        # Test top-level structure
        assert "graphs" in real_go_data
        assert isinstance(real_go_data["graphs"], list)
        assert len(real_go_data["graphs"]) == 1
        
        graph = real_go_data["graphs"][0]
        assert "nodes" in graph
        assert isinstance(graph["nodes"], list)
        assert len(graph["nodes"]) == 4
        
        # Test node structure
        for node in graph["nodes"]:
            assert "id" in node
            assert "lbl" in node
            assert "type" in node
            assert "meta" in node
            
            assert node["type"] == "CLASS"
            assert node["id"].startswith("http://purl.obolibrary.org/obo/GO_")
            assert isinstance(node["meta"], dict)

    def test_configuration_compatibility(self, real_go_data):
        """Test that parsing works with different configuration options."""
        node = real_go_data["graphs"][0]["nodes"][0]
        
        # Test default configuration
        default_parsed = parse_go_term(node)
        assert default_parsed["id"] == "GO:0000001"
        
        # Test custom ID format configuration
        custom_config = {"prefix_replacement": {"_": "-"}}
        custom_parsed = parse_go_term(node, custom_config)
        assert custom_parsed["id"] == "GO-0000001"
        
        # Test no replacement configuration
        no_replace_config = {"prefix_replacement": {}}
        no_replace_parsed = parse_go_term(node, no_replace_config)
        assert no_replace_parsed["id"] == "GO_0000001"

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test node without required fields
        invalid_node = {"id": "test", "meta": {}}
        parsed = parse_go_term(invalid_node)
        assert parsed is None
        
        # Test node without definition
        no_def_node = {
            "id": "http://purl.obolibrary.org/obo/GO_0000001",
            "lbl": "test term",
            "meta": {}
        }
        parsed = parse_go_term(no_def_node)
        assert parsed["definition"] == ""
        
        # Test malformed URI
        malformed_node = {
            "id": "malformed_uri",
            "lbl": "test term",
            "meta": {}
        }
        parsed = parse_go_term(malformed_node)
        assert parsed["id"] == "malformed:uri"