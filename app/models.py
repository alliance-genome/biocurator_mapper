from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class ResolveRequest(BaseModel):
    passage: str
    ontology_name: str = Field(..., description="Ontology short name, e.g. GO")


class OntologyTerm(BaseModel):
    id: str
    name: str
    definition: Optional[str] = None


class ResolveResponse(BaseModel):
    best_match: OntologyTerm | None = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    alternatives: Optional[List[OntologyTerm]] = None
    error: Optional[str] = None


class OntologyUpdateRequest(BaseModel):
    ontology_name: str
    source_url: str


@dataclass
class DOTerm:
    """
    Representation of a Disease Ontology (DO) term.
    
    Attributes:
        id (str): Unique identifier for the DO term
        name (str): Primary name of the term
        definition (Optional[str]): Description of the term
        synonyms (Optional[Dict[str, List[str]]]): Synonyms categorized by type
    """
    id: str
    name: str
    definition: Optional[str] = None
    synonyms: Optional[Dict[str, List[str]]] = field(default_factory=dict)