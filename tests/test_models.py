import pytest
from pydantic import ValidationError

from app.models import ResolveRequest, ResolveResponse, OntologyUpdateRequest, OntologyTerm


class TestModels:
    """Test Pydantic models."""

    def test_resolve_request_valid(self):
        """Test valid ResolveRequest creation."""
        request = ResolveRequest(
            passage="Test passage about cell division",
            ontology_name="GO"
        )
        
        assert request.passage == "Test passage about cell division"
        assert request.ontology_name == "GO"

    def test_resolve_request_missing_passage(self):
        """Test ResolveRequest with missing passage."""
        with pytest.raises(ValidationError):
            ResolveRequest(ontology_name="GO")

    def test_resolve_request_missing_ontology_name(self):
        """Test ResolveRequest with missing ontology_name."""
        with pytest.raises(ValidationError):
            ResolveRequest(passage="Test passage")

    def test_resolve_request_empty_passage(self):
        """Test ResolveRequest with empty passage."""
        # Empty strings are currently allowed by the model
        request = ResolveRequest(passage="", ontology_name="GO")
        assert request.passage == ""
        assert request.ontology_name == "GO"

    def test_ontology_term_valid(self):
        """Test valid OntologyTerm creation."""
        term = OntologyTerm(
            id="GO:0001",
            name="Test Term",
            definition="Test definition"
        )
        
        assert term.id == "GO:0001"
        assert term.name == "Test Term"
        assert term.definition == "Test definition"

    def test_ontology_term_optional_definition(self):
        """Test OntologyTerm with optional definition."""
        term = OntologyTerm(
            id="GO:0001",
            name="Test Term"
        )
        
        assert term.id == "GO:0001"
        assert term.name == "Test Term"
        assert term.definition is None

    def test_ontology_update_request_valid(self):
        """Test valid OntologyUpdateRequest creation."""
        request = OntologyUpdateRequest(
            ontology_name="GO",
            source_url="http://example.com/go.json"
        )
        
        assert request.ontology_name == "GO"
        assert request.source_url == "http://example.com/go.json"

    def test_ontology_update_request_missing_fields(self):
        """Test OntologyUpdateRequest with missing fields."""
        with pytest.raises(ValidationError):
            OntologyUpdateRequest(ontology_name="GO")
        
        with pytest.raises(ValidationError):
            OntologyUpdateRequest(source_url="http://example.com/go.json")

    def test_resolve_response_valid(self):
        """Test valid ResolveResponse creation."""
        best_match = OntologyTerm(id="GO:0001", name="Test Term")
        alternatives = [OntologyTerm(id="GO:0002", name="Alt Term")]
        
        response = ResolveResponse(
            best_match=best_match,
            confidence=0.95,
            reason="High semantic similarity",
            alternatives=alternatives
        )
        
        assert response.best_match.id == "GO:0001"
        assert response.confidence == 0.95
        assert response.reason == "High semantic similarity"
        assert len(response.alternatives) == 1
        assert response.alternatives[0].id == "GO:0002"

    def test_resolve_response_optional_fields(self):
        """Test ResolveResponse with optional fields."""
        best_match = OntologyTerm(id="GO:0001", name="Test Term")
        
        response = ResolveResponse(best_match=best_match)
        
        assert response.best_match.id == "GO:0001"
        assert response.confidence is None
        assert response.reason is None
        assert response.alternatives is None