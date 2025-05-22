#!/usr/bin/env python3
"""
Simple test runner for the biocurator mapper project.
This script runs basic validation tests without requiring pytest installation.
"""

import sys
import os
import yaml
import tempfile
from unittest.mock import patch, mock_open

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_yaml_config_loading():
    """Test that the YAML configuration loads correctly."""
    print("Testing YAML configuration loading...")
    
    try:
        with open('ontology_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'ontologies' in config
        assert 'GO' in config['ontologies']
        assert 'DOID' in config['ontologies']
        print("✓ YAML configuration loads correctly")
        return True
    except Exception as e:
        print(f"✗ YAML configuration test failed: {e}")
        return False

def test_config_module():
    """Test that the config module imports and loads correctly."""
    print("Testing config module import...")
    
    try:
        from config import load_ontology_config, ONTOLOGY_CONFIG
        
        # Test that the config loads
        assert ONTOLOGY_CONFIG is not None
        assert 'ontologies' in ONTOLOGY_CONFIG
        
        # Test the function works
        config = load_ontology_config()
        assert 'ontologies' in config
        
        print("✓ Config module imports and works correctly")
        return True
    except Exception as e:
        print(f"✗ Config module test failed: {e}")
        return False

def test_models_validation():
    """Test that the Pydantic models work correctly."""
    print("Testing Pydantic models...")
    
    try:
        from models import ResolveRequest, OntologyTerm, OntologyUpdateRequest
        
        # Test valid model creation
        request = ResolveRequest(passage="test passage", ontology_name="GO")
        assert request.passage == "test passage"
        assert request.ontology_name == "GO"
        
        term = OntologyTerm(id="GO:0001", name="Test Term")
        assert term.id == "GO:0001"
        assert term.name == "Test Term"
        
        update_req = OntologyUpdateRequest(ontology_name="GO", source_url="http://example.com")
        assert update_req.ontology_name == "GO"
        
        print("✓ Pydantic models work correctly")
        return True
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False

def test_main_helper_functions():
    """Test helper functions in main.py."""
    print("Testing main.py helper functions...")
    
    try:
        from main import get_nested_value, get_ontology_config
        
        # Test get_nested_value
        data = {"meta": {"definition": {"val": "test"}}}
        result = get_nested_value(data, ["meta", "definition", "val"])
        assert result == "test"
        
        result = get_nested_value({}, ["missing", "path"], "default")
        assert result == "default"
        
        print("✓ Helper functions work correctly")
        return True
    except Exception as e:
        print(f"✗ Helper functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running biocurator mapper tests...\n")
    
    tests = [
        test_yaml_config_loading,
        test_config_module,
        test_models_validation,
        test_main_helper_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())